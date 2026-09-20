//! Transport-agnostic daemon request service.
//!
//! Everything between "raw request payload" and "response envelope" lives
//! here: parsing, session resolution, access-tier authorization, dispatch,
//! and error mapping. Transports (the Unix-socket listener today, a network
//! endpoint tomorrow) own only framing and connection lifecycle, and hand
//! each payload to [`handle_request`] together with the [`ConnectionPolicy`]
//! their trust boundary implies.
//!
//! The session tier is a property of the transport, not the request. The
//! policy sets a ceiling; a client-supplied `session` field may restrict a
//! connection below that ceiling but can never raise it. Payload bytes are
//! attacker-controlled on a network transport, so nothing parsed from them
//! participates in granting authority.

use std::path::{Path, PathBuf};
use std::time::Duration;

use serde_json::Value;
use tokio::time::timeout;

use crate::config::HadesConfig;
use crate::db::ArangoPool;
use crate::dispatch::{
    self, AccessTier, DaemonCommand, DaemonResponse, DispatchError, HandlerError,
};

// ---------------------------------------------------------------------------
// Session policy
// ---------------------------------------------------------------------------

/// Whether a session is an AI agent (restricted) or admin (unrestricted).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionKind {
    /// Unrestricted — human operator on a trusted local transport.
    Admin,
    /// Restricted — limited to [`AccessTier::Agent`] commands.
    Agent,
}

/// The authority ceiling a transport grants its connections.
///
/// Constructed by the transport, never from request bytes. A request may
/// self-restrict below the ceiling via its `session` field; requesting a
/// tier above the ceiling is an error, not a silent clamp.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConnectionPolicy {
    /// Highest tier this connection may operate at.
    pub max_tier: SessionKind,
    /// Whether this connection may invoke [`AccessTier::Provisioning`]
    /// commands: creating a database and starting an ingest.
    ///
    /// Off by default and settable only by the transport, never by a request.
    /// A network transport that turns this on is handing a bearer token the
    /// ability to create databases and to make the daemon read paths off the
    /// server's filesystem, so it comes with [`ProvisioningLimits`] rather
    /// than alone.
    pub allow_provisioning: bool,
    /// What a provisioned connection may name. Ignored when
    /// `allow_provisioning` is false.
    pub provisioning: ProvisioningLimits,
}

/// Bounds on what a provisioned connection may create and read.
///
/// Both lists are empty by default, and empty means "nothing is permitted"
/// rather than "everything": a transport that grants provisioning without
/// saying what may be named has granted nothing, which is the safe direction
/// for a default to fail in.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ProvisioningLimits {
    /// Database names a provisioned connection may create must start with one
    /// of these. Keeps a LAN client from creating or targeting a name that
    /// collides with a production graph.
    pub database_prefixes: Vec<String>,
    /// Directories an ingest may read from. A path is permitted only if it
    /// canonicalizes to something inside one of these.
    ///
    /// Without this an MCP client can have the daemon read and embed anything
    /// the daemon's user can read, then query it back out through `db.query`,
    /// which is exfiltration wearing a retrieval interface.
    pub ingest_roots: Vec<PathBuf>,
}

impl ConnectionPolicy {
    /// Policy for trusted local transports (Unix socket peers): admin
    /// ceiling, admin default — the daemon's historical local behavior.
    ///
    /// Provisioning is allowed because an admin session could already create a
    /// database and run an ingest by other means on this transport, and the
    /// limits are empty because they do not gate an admin session.
    pub fn local_admin() -> Self {
        Self {
            max_tier: SessionKind::Admin,
            allow_provisioning: true,
            provisioning: ProvisioningLimits::default(),
        }
    }

    /// Policy for network transports: agent ceiling. A request claiming
    /// an admin session is rejected outright.
    pub fn agent_only() -> Self {
        Self {
            max_tier: SessionKind::Agent,
            allow_provisioning: false,
            provisioning: ProvisioningLimits::default(),
        }
    }

    /// Agent ceiling, plus provisioning inside `limits`.
    ///
    /// The ceiling stays at agent on purpose: this grants exactly the
    /// provisioning commands and nothing else, so raw AQL, purge and graph drop
    /// remain unavailable to the transport that gets this.
    pub fn agent_with_provisioning(limits: ProvisioningLimits) -> Self {
        Self {
            max_tier: SessionKind::Agent,
            allow_provisioning: true,
            provisioning: limits,
        }
    }

    /// Whether `name` is a database this connection may create or provision.
    pub fn permits_database(&self, name: &str) -> bool {
        if !self.allow_provisioning {
            return false;
        }
        if self.max_tier == SessionKind::Admin && self.provisioning.database_prefixes.is_empty() {
            return true;
        }
        // An empty prefix is a prefix of everything, so one blank entry would
        // turn this gate and the MCP read allowlist into allow-all. A blank
        // comes from a bare `--mcp-db-prefix=` or an empty
        // `HADES_MCP_DB_PREFIXES=` in an environment file, both of which clap's
        // comma delimiter turns into one empty value. Rejected at startup as
        // well; ignored here so the two cannot disagree.
        self.provisioning
            .database_prefixes
            .iter()
            .filter(|prefix| !prefix.is_empty())
            .any(|prefix| name.starts_with(prefix))
    }

    /// Whether `path` is inside a directory this connection may ingest from.
    ///
    /// Canonicalizes first, so `..` and symlinks cannot walk out of a permitted
    /// root. A path that does not exist is refused: there is nothing to ingest,
    /// and resolving a missing path would have to guess.
    pub fn permits_ingest_path(&self, path: &Path) -> bool {
        if !self.allow_provisioning {
            return false;
        }
        if self.max_tier == SessionKind::Admin && self.provisioning.ingest_roots.is_empty() {
            return true;
        }
        let Ok(candidate) = std::fs::canonicalize(path) else {
            return false;
        };
        self.provisioning.ingest_roots.iter().any(|root| {
            std::fs::canonicalize(root)
                .map(|root| candidate.starts_with(root))
                .unwrap_or(false)
        })
    }
}

/// Resolve the effective session tier from the transport policy and the
/// request's optional self-declared session.
///
/// Absent → the policy ceiling (admin locally, agent on the network).
/// `"agent"` → agent, always honored (self-restriction).
/// `"admin"` → admin only when the ceiling allows it; otherwise
/// `ACCESS_DENIED`, because silently downgrading an explicit claim would
/// misreport what tier the commands then run at.
pub fn resolve_session(
    policy: &ConnectionPolicy,
    requested: Option<SessionKind>,
    request_id: &Option<String>,
) -> Result<SessionKind, DaemonResponse> {
    match requested {
        None => Ok(policy.max_tier),
        Some(SessionKind::Agent) => Ok(SessionKind::Agent),
        Some(SessionKind::Admin) => match policy.max_tier {
            SessionKind::Admin => Ok(SessionKind::Admin),
            SessionKind::Agent => Err(DaemonResponse::err(
                "ACCESS_DENIED",
                "transport policy caps this connection at agent tier; \
                 admin session is not available on this endpoint",
            )
            .with_request_id(request_id.clone())),
        },
    }
}

// ---------------------------------------------------------------------------
// Request parsing
// ---------------------------------------------------------------------------

/// A parsed request frame, before authorization.
#[derive(Debug)]
pub struct ParsedRequest {
    /// Client-chosen correlation id, echoed on every response.
    pub request_id: Option<String>,
    /// Self-declared session from the frame, if present. Advisory only —
    /// [`resolve_session`] decides what it is worth.
    pub requested_session: Option<SessionKind>,
    /// The decoded command.
    pub command: DaemonCommand,
}

/// Parse a raw JSON payload into a [`ParsedRequest`].
///
/// Returns `Err(DaemonResponse)` on failure — the caller sends it directly
/// to the client.
pub fn parse_request(payload: &[u8]) -> Result<ParsedRequest, DaemonResponse> {
    let frame: Value = serde_json::from_slice(payload)
        .map_err(|e| DaemonResponse::err("MALFORMED_JSON", e.to_string()))?;

    let request_id = frame
        .get("request_id")
        .and_then(|v| v.as_str())
        .map(String::from);

    let requested_session = match frame.get("session").and_then(|v| v.as_str()) {
        Some("agent") => Some(SessionKind::Agent),
        Some("admin") => Some(SessionKind::Admin),
        None => None,
        Some(other) => {
            return Err(DaemonResponse::err(
                "INVALID_SESSION",
                format!("unknown session type '{other}'; expected \"agent\" or \"admin\""),
            )
            .with_request_id(request_id));
        }
    };

    let command: DaemonCommand = serde_json::from_value(frame).map_err(|e| {
        DaemonResponse::err("UNKNOWN_COMMAND", e.to_string()).with_request_id(request_id.clone())
    })?;

    Ok(ParsedRequest {
        request_id,
        requested_session,
        command,
    })
}

/// Render a prefix list for an error message, saying so when it is empty.
fn describe_prefixes(prefixes: &[String]) -> String {
    if prefixes.is_empty() {
        "none configured, so no database may be created here".to_string()
    } else {
        prefixes.join(", ")
    }
}

/// Render a root list for an error message, saying so when it is empty.
fn describe_roots(roots: &[PathBuf]) -> String {
    if roots.is_empty() {
        "none configured, so nothing may be ingested here".to_string()
    } else {
        roots
            .iter()
            .map(|r| r.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    }
}

// ---------------------------------------------------------------------------
// Request handling
// ---------------------------------------------------------------------------

/// Handle one request payload end to end: parse, resolve the session tier
/// against the transport policy, enforce access tiers, dispatch with a
/// timeout, and map errors to the protocol envelope.
///
/// Always returns a [`DaemonResponse`]; transport-level concerns (framing,
/// payload size limits, idle timeouts) belong to the caller.
pub async fn handle_request(
    pool: &ArangoPool,
    config: &HadesConfig,
    policy: ConnectionPolicy,
    payload: &[u8],
    request_timeout: Duration,
) -> DaemonResponse {
    let parsed = match parse_request(payload) {
        Ok(parsed) => parsed,
        Err(resp) => return resp,
    };

    let session = match resolve_session(&policy, parsed.requested_session, &parsed.request_id) {
        Ok(session) => session,
        Err(resp) => return resp,
    };

    let request_id = parsed.request_id;
    let cmd = parsed.command;

    // Enforce access tier. Agent-tier commands are always available; Admin and
    // Internal need an admin session; Provisioning needs an admin session or a
    // transport that explicitly granted it. The grant comes from the policy the
    // transport constructed, so a request still cannot raise its own authority.
    let permitted = match cmd.access_tier() {
        AccessTier::Agent => true,
        // Provisioning requires the transport to have granted it AND the session
        // to still be at the level the transport grants. `||` on the policy flag
        // ignored self-restriction: on the Unix socket, where the ceiling is
        // admin and provisioning comes with it, a client declaring
        // `"session":"agent"` resolved to Agent and was then let through on the
        // transport flag alone. The documented guarantee is that a request may
        // restrict itself below the ceiling, and it has to hold for this tier.
        AccessTier::Provisioning => policy.allow_provisioning && session == policy.max_tier,
        AccessTier::Admin | AccessTier::Internal => session == SessionKind::Admin,
    };
    if !permitted {
        return DaemonResponse::err(
            "ACCESS_DENIED",
            format!(
                "{:?} session cannot invoke {:?}-tier command on this endpoint",
                session,
                cmd.access_tier(),
            ),
        )
        .with_request_id(request_id);
    }

    // Provisioning limits. The tier check above says the connection may
    // provision; these say what it may name. Enforced here rather than in the
    // handler because it is authorization, and authorization belongs beside the
    // policy that granted it, not beside the operation that obeys it.
    match &cmd {
        DaemonCommand::DbCreateDatabase(p) if !policy.permits_database(&p.name) => {
            return DaemonResponse::err(
                "ACCESS_DENIED",
                format!(
                    "this endpoint may not create database '{}'. Permitted prefixes: {}",
                    p.name,
                    describe_prefixes(&policy.provisioning.database_prefixes),
                ),
            )
            .with_request_id(request_id);
        }
        DaemonCommand::IngestStart(p) if !policy.permits_ingest_path(Path::new(&p.path)) => {
            return DaemonResponse::err(
                "ACCESS_DENIED",
                format!(
                    "this endpoint may not ingest '{}'. It must exist and lie inside \
                     one of the permitted roots: {}",
                    p.path,
                    describe_roots(&policy.provisioning.ingest_roots),
                ),
            )
            .with_request_id(request_id);
        }
        _ => {}
    }

    // Preserve command payload for NotImplemented subprocess fallback.
    let cmd_value = serde_json::to_value(&cmd).ok();

    match timeout(request_timeout, dispatch::dispatch(pool, config, cmd)).await {
        Ok(Ok(data)) => DaemonResponse::ok(data).with_request_id(request_id),
        Ok(Err(DispatchError::NotImplemented(name))) => {
            let mut resp = DaemonResponse::err(
                "NOT_IMPLEMENTED",
                format!("command '{name}' not yet implemented natively"),
            );
            resp.data = cmd_value;
            resp.with_request_id(request_id)
        }
        Ok(Err(DispatchError::Handler(e))) => {
            DaemonResponse::err(handler_error_code(&e), error_chain(&e)).with_request_id(request_id)
        }
        Err(_) => DaemonResponse::err("INTERNAL", "request timed out").with_request_id(request_id),
    }
}

/// Map a [`HandlerError`] to a protocol error code string.
pub fn handler_error_code(e: &HandlerError) -> &'static str {
    match e {
        HandlerError::InvalidNodeId { .. }
        | HandlerError::InvalidLimit { .. }
        | HandlerError::InvalidParameter { .. } => "INVALID_PARAMS",
        HandlerError::NodeNotFound(_) | HandlerError::DocumentNotFound { .. } => "NOT_FOUND",
        HandlerError::NoEmbedding { .. } | HandlerError::InvalidEmbedding { .. } => "QUERY_FAILED",
        // A query that failed because something was not there is a different
        // answer from a query that broke, and a caller surveying a graph needs
        // to tell them apart: "no such collection" ends the question, while
        // QUERY_FAILED invites a retry. Reported by one session as turning a
        // single question into six probes to establish a negative.
        HandlerError::Query { source, .. } if source.is_not_found() => "NOT_FOUND",
        HandlerError::Query { .. } => "QUERY_FAILED",
        HandlerError::InsertFailed { .. } => "INSERT_FAILED",
        HandlerError::MaterializationFailed { .. } => "MATERIALIZATION_FAILED",
        HandlerError::SearchOverloaded => "SEARCH_OVERLOADED",
        HandlerError::ServiceError(_) => "SERVICE_ERROR",
    }
}

/// An error and everything under it, as one line.
///
/// `HandlerError::Query` carries its `ArangoError` as `#[source]` and renders
/// only its own context, so building a response from `to_string()` dropped the
/// reason at the last step: `graph_neighbors` reported "graph traverse from
/// 'codebase_files/x'", which restates the request, while the cause it already
/// held said `graph 'default' not found`. Walking the chain costs nothing and
/// turns a restatement into an answer.
pub fn error_chain(e: &(dyn std::error::Error + 'static)) -> String {
    let mut message = e.to_string();
    let mut cause = e.source();
    while let Some(next) = cause {
        let text = next.to_string();
        // thiserror's `#[error(transparent)]` repeats the source verbatim, and a
        // message ending in its own cause reads worse for the repetition.
        if !message.contains(&text) {
            message.push_str(": ");
            message.push_str(&text);
        }
        cause = next.source();
    }
    message
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
use crate::cursor_mock;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handler_error_code_mapping() {
        assert_eq!(
            handler_error_code(&HandlerError::NodeNotFound("x".into())),
            "NOT_FOUND",
        );
        assert_eq!(
            handler_error_code(&HandlerError::InvalidNodeId {
                node_id: "x".into(),
                reason: "bad".into(),
            }),
            "INVALID_PARAMS",
        );
        assert_eq!(
            handler_error_code(&HandlerError::InvalidLimit {
                limit: 0,
                max: 1000
            }),
            "INVALID_PARAMS",
        );
        assert_eq!(
            handler_error_code(&HandlerError::NoEmbedding {
                node_id: "x".into(),
            }),
            "QUERY_FAILED",
        );
    }

    #[test]
    fn parse_rejects_malformed_json() {
        let resp = parse_request(b"not json at all").unwrap_err();
        assert!(!resp.success);
        assert_eq!(resp.error_code.as_deref(), Some("MALFORMED_JSON"));
    }

    #[test]
    fn parse_rejects_unknown_command() {
        let payload = serde_json::to_vec(&serde_json::json!({
            "command": "nonexistent.cmd",
            "params": {}
        }))
        .unwrap();
        let resp = parse_request(&payload).unwrap_err();
        assert!(!resp.success);
        assert_eq!(resp.error_code.as_deref(), Some("UNKNOWN_COMMAND"));
    }

    #[test]
    fn parse_echoes_request_id_on_error() {
        let payload = serde_json::to_vec(&serde_json::json!({
            "request_id": "req-42",
            "command": "nonexistent.cmd",
            "params": {}
        }))
        .unwrap();
        let resp = parse_request(&payload).unwrap_err();
        assert_eq!(resp.request_id.as_deref(), Some("req-42"));
    }

    #[test]
    fn parse_leaves_absent_session_unresolved() {
        let payload = serde_json::to_vec(&serde_json::json!({
            "request_id": "req-1",
            "command": "orient",
            "params": {}
        }))
        .unwrap();
        let parsed = parse_request(&payload).unwrap();
        assert_eq!(parsed.request_id.as_deref(), Some("req-1"));
        // Absent session is None — the transport policy decides the tier,
        // not the parser.
        assert_eq!(parsed.requested_session, None);
    }

    #[test]
    fn parse_reads_explicit_agent_session() {
        let payload = serde_json::to_vec(&serde_json::json!({
            "session": "agent",
            "command": "orient",
            "params": {}
        }))
        .unwrap();
        let parsed = parse_request(&payload).unwrap();
        assert_eq!(parsed.requested_session, Some(SessionKind::Agent));
    }

    #[test]
    fn parse_rejects_unknown_session() {
        let payload = serde_json::to_vec(&serde_json::json!({
            "session": "superuser",
            "command": "orient",
            "params": {}
        }))
        .unwrap();
        let resp = parse_request(&payload).unwrap_err();
        assert!(!resp.success);
        assert_eq!(resp.error_code.as_deref(), Some("INVALID_SESSION"));
    }

    // --- resolve_session: the A1 matrix -----------------------------------

    #[test]
    fn local_admin_policy_defaults_to_admin() {
        // Historical local behavior: absent session on the Unix socket → admin.
        let tier = resolve_session(&ConnectionPolicy::local_admin(), None, &None).unwrap();
        assert_eq!(tier, SessionKind::Admin);
    }

    #[test]
    fn local_admin_policy_honors_self_restriction() {
        let tier = resolve_session(
            &ConnectionPolicy::local_admin(),
            Some(SessionKind::Agent),
            &None,
        )
        .unwrap();
        assert_eq!(tier, SessionKind::Agent);
    }

    #[test]
    fn agent_only_policy_defaults_to_agent() {
        // Absent session on a network transport → agent, never admin.
        let tier = resolve_session(&ConnectionPolicy::agent_only(), None, &None).unwrap();
        assert_eq!(tier, SessionKind::Agent);
    }

    #[test]
    fn agent_only_policy_rejects_admin_claim() {
        // The A1 fix: request bytes cannot elevate above the transport ceiling.
        let resp = resolve_session(
            &ConnectionPolicy::agent_only(),
            Some(SessionKind::Admin),
            &Some("req-9".into()),
        )
        .unwrap_err();
        assert!(!resp.success);
        assert_eq!(resp.error_code.as_deref(), Some("ACCESS_DENIED"));
        assert_eq!(resp.request_id.as_deref(), Some("req-9"));
    }

    #[test]
    fn agent_only_policy_accepts_explicit_agent() {
        let tier = resolve_session(
            &ConnectionPolicy::agent_only(),
            Some(SessionKind::Agent),
            &None,
        )
        .unwrap();
        assert_eq!(tier, SessionKind::Agent);
    }

    // --- handle_request: authorization before dispatch --------------------

    #[tokio::test]
    async fn agent_session_blocked_from_admin_command() {
        // Tier check fires before dispatch — no live DB needed.
        let mut config = HadesConfig::default();
        config.database.name = Some("fixture".to_string());
        let mut mock = cursor_mock::Mock::new(vec![cursor_mock::Reply::page(
            serde_json::json!({"result":[7],"hasMore":false}),
        )])
        .await;

        let payload = serde_json::to_vec(&serde_json::json!({
            "session": "agent",
            "command": "db.aql",
            "params": { "aql": "RETURN 1" }
        }))
        .unwrap();

        let resp = handle_request(
            &mock.pool,
            &config,
            ConnectionPolicy::local_admin(),
            &payload,
            Duration::from_secs(5),
        )
        .await;

        assert!(!resp.success);
        assert_eq!(resp.error_code.as_deref(), Some("ACCESS_DENIED"));
        assert!(
            mock.events.try_recv().is_err(),
            "rejected command reached database mock"
        );
    }

    #[tokio::test]
    async fn agent_only_policy_blocks_admin_command_without_session_field() {
        // The frame carries no session at all — under an agent-only policy
        // the admin-tier command must still be denied. This is the exact
        // payload that reached dispatch as Admin before the A1 fix.
        let mut config = HadesConfig::default();
        config.database.name = Some("fixture".to_string());
        let mut mock = cursor_mock::Mock::new(vec![cursor_mock::Reply::page(
            serde_json::json!({"result":[7],"hasMore":false}),
        )])
        .await;

        let payload = serde_json::to_vec(&serde_json::json!({
            "command": "db.aql",
            "params": { "aql": "RETURN 1" }
        }))
        .unwrap();

        let resp = handle_request(
            &mock.pool,
            &config,
            ConnectionPolicy::agent_only(),
            &payload,
            Duration::from_secs(5),
        )
        .await;

        assert!(!resp.success);
        assert_eq!(resp.error_code.as_deref(), Some("ACCESS_DENIED"));
        assert!(
            mock.events.try_recv().is_err(),
            "rejected command reached database mock"
        );
    }

    // ── Provisioning limits ─────────────────────────────────────────────

    #[test]
    fn an_error_reports_its_cause_not_a_restatement() {
        let e = HandlerError::Query {
            context: "graph traverse from 'codebase_files/x'".into(),
            source: crate::db::ArangoError::Request("graph 'default' not found".into()),
        };
        let rendered = error_chain(&e);
        assert!(
            rendered.contains("graph 'default' not found"),
            "the cause must survive: {rendered}"
        );
        assert!(
            rendered.contains("graph traverse from"),
            "the context must survive too: {rendered}"
        );
    }

    #[test]
    fn a_plain_network_endpoint_provisions_nothing() {
        let policy = ConnectionPolicy::agent_only();
        assert!(!policy.permits_database("bident_v5"));
        assert!(!policy.permits_ingest_path(Path::new("/opt/HADES")));
    }

    #[test]
    fn provisioning_without_limits_permits_nothing_either() {
        // Empty lists mean nothing is in bounds, not everything. A transport
        // that grants the tier and forgets the bounds has granted no reach.
        let policy = ConnectionPolicy::agent_with_provisioning(ProvisioningLimits::default());
        assert!(!policy.permits_database("bident_v5"));
        assert!(!policy.permits_ingest_path(Path::new("/opt/HADES")));
    }

    #[test]
    fn database_names_are_gated_by_prefix() {
        let policy = ConnectionPolicy::agent_with_provisioning(ProvisioningLimits {
            database_prefixes: vec!["bident_".to_string()],
            ingest_roots: Vec::new(),
        });
        assert!(policy.permits_database("bident_v5"));
        assert!(!policy.permits_database("WeaverTools_v4"));
        assert!(!policy.permits_database("_system"));
    }

    #[test]
    fn ingest_paths_cannot_walk_out_of_a_permitted_root() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path().join("allowed");
        let outside = tmp.path().join("secrets");
        std::fs::create_dir_all(root.join("inner")).expect("mkdir");
        std::fs::create_dir_all(&outside).expect("mkdir");
        std::fs::write(outside.join("token"), "s3cret").expect("write");

        let policy = ConnectionPolicy::agent_with_provisioning(ProvisioningLimits {
            database_prefixes: vec!["bident_".to_string()],
            ingest_roots: vec![root.clone()],
        });

        assert!(policy.permits_ingest_path(&root));
        assert!(policy.permits_ingest_path(&root.join("inner")));

        // The whole point: a traversal that resolves outside the root is refused
        // rather than read, because ingest turns a readable path into queryable
        // vectors.
        assert!(!policy.permits_ingest_path(&outside));
        assert!(!policy.permits_ingest_path(&root.join("../secrets")));
        assert!(!policy.permits_ingest_path(&root.join("../secrets/token")));

        // A path that does not exist cannot be shown to be inside the root, so
        // it is refused rather than guessed at.
        assert!(!policy.permits_ingest_path(&root.join("nope")));
    }

    /// A blank entry is a prefix of every name, so it cannot be treated as one.
    #[test]
    fn an_empty_prefix_does_not_permit_everything() {
        let policy = ConnectionPolicy::agent_with_provisioning(ProvisioningLimits {
            database_prefixes: vec![String::new()],
            ingest_roots: vec![std::path::PathBuf::from("/opt")],
        });
        assert!(!policy.permits_database("_system"));
        assert!(!policy.permits_database("WeaverTools_v4"));
    }

    /// Self-restriction has to hold for the provisioning tier as well: the Unix
    /// socket grants it with the admin ceiling, and a client that declares
    /// `"session":"agent"` has asked not to have it.
    #[test]
    fn a_self_restricted_session_loses_provisioning() {
        let local = ConnectionPolicy::local_admin();
        assert!(local.allow_provisioning);

        // What handle_request computes, for each resolved session.
        let permitted = |policy: &ConnectionPolicy, session: SessionKind| {
            policy.allow_provisioning && session == policy.max_tier
        };
        assert!(permitted(&local, SessionKind::Admin));
        assert!(
            !permitted(&local, SessionKind::Agent),
            "declaring an agent session must give up provisioning"
        );

        // The network endpoint's granted level IS agent, so it keeps it there.
        let network = ConnectionPolicy::agent_with_provisioning(ProvisioningLimits {
            database_prefixes: vec!["bident_".to_string()],
            ingest_roots: vec![std::path::PathBuf::from("/opt")],
        });
        assert!(permitted(&network, SessionKind::Agent));
    }

    #[test]
    fn a_local_admin_transport_is_not_narrowed_by_empty_limits() {
        // The Unix socket is a local operator who could already do all of this
        // with the CLI, so empty limits there mean unrestricted rather than
        // nothing. Only the network path fails closed.
        let policy = ConnectionPolicy::local_admin();
        assert!(policy.permits_database("anything"));
        assert!(policy.permits_ingest_path(Path::new("/")));
    }
}

#[cfg(test)]
mod insert_outcome_tests {
    use super::*;
    use serde_json::{Value, json};

    async fn insert(data: Value, response: Value) -> DaemonResponse {
        let mut mock = cursor_mock::Mock::new(vec![cursor_mock::Reply::page(response)]).await;
        let payload =
            serde_json::to_vec(&json!({"request_id":"insert-test", "command":"db.insert",
            "params":{"collection":"docs", "data":data}}))
            .unwrap();
        let result = handle_request(
            &mock.pool,
            &HadesConfig::default(),
            ConnectionPolicy::local_admin(),
            &payload,
            Duration::from_secs(3),
        )
        .await;
        assert_eq!(mock.event("POST document/docs").await, data);
        assert_eq!(result.request_id.as_deref(), Some("insert-test"));
        assert!(
            mock.events.try_recv().is_err(),
            "insert must not retry automatically"
        );
        result
    }
    fn good() -> Value {
        json!({"_id":"docs/new", "_key":"new", "_rev":"revision"})
    }
    fn bad() -> Value {
        json!({"error":true,"errorNum":1210,"errorMessage":"duplicate fixture key"})
    }

    #[tokio::test]
    async fn insert_failure_envelope_preserves_partial_outcome_diagnostics() {
        for (responses, expected) in [
            (json!([good(), bad()]), "1 succeeded, 1 failed"),
            (json!([bad(), bad()]), "0 succeeded, 2 failed"),
        ] {
            let result = insert(json!([{}, {}]), responses).await;
            assert!(!result.success);
            assert_eq!(result.error_code.as_deref(), Some("INSERT_FAILED"));
            assert!(result.data.is_none());
            let error = result.error.unwrap();
            assert!(error.contains(expected), "{error}");
            assert!(error.contains("1210") && error.contains("duplicate fixture key"));
            assert!(error.contains("successful items remain committed"));
            assert!(error.contains("\"index\":1"));
        }
    }

    #[tokio::test]
    async fn insert_success_preserves_single_and_array_payloads() {
        for (input, output) in [
            (json!({"error":true}), good()),
            (json!([{}, {}]), json!([good(), good()])),
            (json!([]), json!([])),
        ] {
            let result = insert(input, output.clone()).await;
            assert!(result.success, "{:?}", result);
            assert_eq!(result.data, Some(output));
            assert!(result.error.is_none());
        }
    }

    #[tokio::test]
    async fn malformed_insert_responses_do_not_claim_known_outcomes() {
        for response in [
            json!({}),
            json!([good()]),
            json!([good(), null]),
            json!([good(), {}]),
            json!([good(),{"error":"true"}]),
        ] {
            let result = insert(json!([{}, {}]), response).await;
            assert!(!result.success);
            assert_eq!(result.error_code.as_deref(), Some("QUERY_FAILED"));
            assert!(result.error.unwrap().contains("write outcome is uncertain"));
        }
        let result = insert(json!({}), json!([])).await;
        assert!(!result.success);
    }

    #[tokio::test]
    async fn invalid_insert_input_is_rejected_before_any_write() {
        for input in [json!(null), json!("text"), json!([{}, 1])] {
            let mut mock = cursor_mock::Mock::new(vec![]).await;
            let payload = serde_json::to_vec(
                &json!({"command":"db.insert","params":{"collection":"docs","data":input}}),
            )
            .unwrap();
            let result = handle_request(
                &mock.pool,
                &HadesConfig::default(),
                ConnectionPolicy::local_admin(),
                &payload,
                Duration::from_secs(3),
            )
            .await;
            assert!(!result.success);
            assert_eq!(result.error_code.as_deref(), Some("INVALID_PARAMS"));
            assert!(mock.events.try_recv().is_err());
        }
    }
}

#[cfg(test)]
mod materialization_outcome_tests {
    use super::*;
    use serde_json::json;

    fn page(value: Value) -> cursor_mock::Reply {
        cursor_mock::Reply::page(value)
    }
    fn failure() -> cursor_mock::Reply {
        cursor_mock::Reply {
            status: 503,
            body: json!({"error":true,"errorNum":503,"errorMessage":"injected failure"}),
            gate: None,
        }
    }
    fn setup(strategy: &str, existing_edge: bool) -> Vec<cursor_mock::Reply> {
        let mut collections = vec![
            json!({"name":"hades_schema","type":2}),
            json!({"name":"docs","type":2}),
        ];
        if existing_edge {
            collections.push(json!({"name":"edges","type":3}));
        }
        vec![
            page(json!({"result":collections})),
            page(json!({"hasMore":false,"result":[
                {"schema_type":"schema_meta"},
                {"schema_type":"edge_definition","name":"edges","source_field":"related","from_collections":["docs"],"to_collections":["docs"],"materialize_strategy":strategy},
                {"schema_type":"named_graph","name":"fixture_graph","edge_definitions":["edges"]}
            ]})),
            page(json!({"result":collections})),
        ]
    }
    async fn execute(
        replies: Vec<cursor_mock::Reply>,
        dry_run: bool,
        register: bool,
    ) -> DaemonResponse {
        let mock = cursor_mock::Mock::new(replies).await;
        let payload = serde_json::to_vec(&json!({"request_id":"materialize-test","command":"db.graph.materialize","params":{"dry_run":dry_run,"register":register}})).unwrap();
        let response = handle_request(
            &mock.pool,
            &HadesConfig::default(),
            ConnectionPolicy::local_admin(),
            &payload,
            Duration::from_secs(3),
        )
        .await;
        assert_eq!(response.request_id.as_deref(), Some("materialize-test"));
        response
    }
    fn report(response: DaemonResponse) -> Value {
        assert!(!response.success, "{response:?}");
        assert!(response.data.is_none());
        assert_eq!(
            response.error_code.as_deref(),
            Some("MATERIALIZATION_FAILED")
        );
        let error = response.error.unwrap();
        assert!(error.contains("earlier writes, if any, remain committed"));
        serde_json::from_str(
            error
                .split_once("inspect report before retrying: ")
                .unwrap()
                .1,
        )
        .unwrap()
    }
    fn source() -> cursor_mock::Reply {
        page(json!({"hasMore":false,"result":[{"_id":"docs/start","ref":"docs/end"}]}))
    }
    #[tokio::test]
    async fn scan_and_strategy_errors_fail_including_dry_run() {
        for dry_run in [false, true] {
            let mut replies = setup("standard", true);
            replies.push(failure());
            let r = report(execute(replies, dry_run, false).await);
            assert_eq!(r["dry_run"], dry_run);
            assert_eq!(r["totals"]["collections_scanned"], 1);
            assert_eq!(r["totals"]["errors"].as_array().unwrap().len(), 1);
        }
        let r = report(execute(setup("unsupported", true), false, false).await);
        assert!(
            r["totals"]["errors"][0]
                .as_str()
                .unwrap()
                .contains("unknown materialize_strategy")
        );
    }
    #[tokio::test]
    async fn import_creation_and_registration_errors_retain_outcomes() {
        let mut replies = setup("standard", true);
        replies.push(source());
        replies.push(page(json!({"created":0,"updated":0,"errors":1})));
        let r = report(execute(replies, false, false).await);
        assert_eq!(r["totals"]["edges_created"], 0);
        assert!(
            r["totals"]["errors"][0]
                .as_str()
                .unwrap()
                .contains("1 of 1 documents failed")
        );
        let mut replies = setup("standard", false);
        replies.extend([source(), failure(), failure()]);
        let r = report(execute(replies, false, false).await);
        assert_eq!(r["totals"]["errors"].as_array().unwrap().len(), 2);
        let mut replies = setup("standard", true);
        replies.extend([source(), page(json!({"created":1,"errors":0})), failure()]);
        let r = report(execute(replies, false, true).await);
        assert_eq!(r["totals"]["edges_created"], 1);
        assert!(
            r["totals"]["errors"][0]
                .as_str()
                .unwrap()
                .contains("create graph fixture_graph")
        );
    }
    #[tokio::test]
    async fn missing_collections_and_skipped_references_are_not_execution_failures() {
        let mut replies = setup("standard", true);
        replies[2].body = json!({"result":[{"name":"hades_schema","type":2}]});
        let response = execute(replies, false, false).await;
        assert!(response.success, "{response:?}");
        assert_eq!(response.data.unwrap()["totals"]["collections_missing"], 1);
        let mut replies = setup("standard", true);
        replies.push(page(
            json!({"hasMore":false,"result":[{"_id":"docs/start","ref":"bare-key"}]}),
        ));
        let response = execute(replies, false, false).await;
        assert!(response.success, "{response:?}");
        assert_eq!(response.data.unwrap()["totals"]["edges_skipped"], 1);
    }

    #[tokio::test]
    async fn successful_materialization_and_dry_run_preserve_report() {
        for dry_run in [false, true] {
            let mut replies = setup("standard", true);
            replies.push(source());
            if !dry_run {
                replies.extend([
                    page(json!({"created":1,"errors":0})),
                    page(json!({"error":false})),
                ]);
            }
            let response = execute(replies, dry_run, true).await;
            assert!(response.success, "{response:?}");
            let r = response.data.unwrap();
            assert_eq!(r["totals"]["edges_created"], 1);
            assert_eq!(r["totals"]["errors"], json!([]));
            assert_eq!(
                r["named_graphs_registered"],
                if dry_run {
                    json!([])
                } else {
                    json!(["fixture_graph"])
                }
            );
        }
    }
}

#[cfg(test)]
mod graph_response_tests {
    use super::*;
    use serde_json::json;

    async fn request(
        command: &str,
        params: Value,
        replies: Vec<Value>,
    ) -> (DaemonResponse, cursor_mock::Mock) {
        let mock =
            cursor_mock::Mock::new(replies.into_iter().map(cursor_mock::Reply::page).collect())
                .await;
        let payload = serde_json::to_vec(
            &json!({"request_id":"graph-test", "command":command,"params":params}),
        )
        .unwrap();
        let response = handle_request(
            &mock.pool,
            &HadesConfig::default(),
            ConnectionPolicy::local_admin(),
            &payload,
            Duration::from_secs(3),
        )
        .await;
        assert_eq!(response.request_id.as_deref(), Some("graph-test"));
        (response, mock)
    }

    #[tokio::test]
    async fn malformed_graph_metadata_fails_listing_and_automatic_resolution() {
        for command in ["db.graph.list", "db.graph.neighbors"] {
            let (response, mut mock) = request(
                command,
                json!({"vertex":"docs/start","direction":"any"}),
                vec![json!({})],
            )
            .await;
            assert!(!response.success, "{response:?}");
            assert_eq!(response.error_code.as_deref(), Some("QUERY_FAILED"));
            assert!(response.data.is_none());
            assert!(response.error.unwrap().contains("graphs must be an array"));
            mock.event("GET gharial").await;
            assert!(
                mock.events.try_recv().is_err(),
                "malformed discovery must not issue a traversal"
            );
        }
    }

    #[tokio::test]
    async fn automatic_graph_selection_preserves_absence_ambiguity_and_single_graph() {
        for (graphs, message) in [
            (json!([]), "no named graph"),
            (
                json!([{"_key":"one","edgeDefinitions":[]},{"_key":"two","edgeDefinitions":[]}]),
                "2 named graphs",
            ),
        ] {
            let (response, mut mock) = request(
                "db.graph.neighbors",
                json!({"vertex":"docs/start","direction":"any"}),
                vec![json!({"graphs":graphs})],
            )
            .await;
            assert!(!response.success, "{response:?}");
            assert!(response.error.unwrap().contains(message));
            mock.event("GET gharial").await;
            assert!(mock.events.try_recv().is_err());
        }
        let (response, mut mock) = request(
            "db.graph.neighbors",
            json!({"vertex":"docs/start","direction":"any"}),
            vec![
                json!({"graphs":[{"_key":"only_graph","edgeDefinitions":[]}]}),
                json!({"result":[],"hasMore":false}),
            ],
        )
        .await;
        assert!(response.success, "{response:?}");
        mock.event("GET gharial").await;
        let query = mock.event("POST cursor").await;
        assert_eq!(query["bindVars"]["graph"], "only_graph");
        assert!(mock.events.try_recv().is_err());
    }
}
