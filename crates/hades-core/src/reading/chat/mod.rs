//! Document-bound conversations and an at-most-once dispatch journal (#214).
mod runtime;
use super::*;
use runtime::Worker;
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex as StdMutex, OnceLock, Weak},
    time::Duration,
};
use tokio::sync::{Mutex, OwnedMutexGuard, Semaphore, SemaphorePermit};

const BINDINGS: &str = "hades_reading_conversations";
const EXCHANGES: &str = "hades_reading_exchanges";
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ChatConfig {
    pub enabled: bool,
    pub executable: String,
    pub home: String,
    pub cwd: String,
    pub mcp_url: String,
    pub mcp_token_env: String,
}
impl Default for ChatConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            executable: "codex".into(),
            home: String::new(),
            cwd: String::new(),
            mcp_url: String::new(),
            mcp_token_env: "HADES_READING_MCP_TOKEN".into(),
        }
    }
}
impl ChatConfig {
    fn validate(&self) -> Result<(), HandlerError> {
        if !self.enabled {
            return Err(failure(
                "Codex reading is not configured; enable reading_chat in the daemon configuration",
            ));
        }
        for path in [&self.home, &self.cwd] {
            if !std::path::Path::new(path).is_absolute() || !std::path::Path::new(path).is_dir() {
                return Err(failure(
                    "Codex home and working directory must be existing absolute directories",
                ));
            }
        }
        let url = url::Url::parse(&self.mcp_url)
            .map_err(|_| failure("Invalid guided-reading MCP URL"))?;
        if url.scheme() != "http"
            || !matches!(url.host_str(), Some("127.0.0.1" | "localhost" | "[::1]"))
            || url.path() != "/mcp"
            || !url.username().is_empty()
            || url.password().is_some()
        {
            return Err(failure(
                "Use a local guided-reading HTTP MCP endpoint ending in /mcp",
            ));
        }
        if std::env::var(&self.mcp_token_env).is_err() {
            return Err(failure(
                "Guided-reading MCP token environment variable is missing",
            ));
        }
        Ok(())
    }
    fn identity(&self) -> String {
        record_hash(&serde_json::to_value(self).unwrap())
    }
}
fn failure(message: impl Into<String>) -> HandlerError {
    HandlerError::ServiceError(message.into())
}
/// Stream transactions return conflicts only before commit; uncertain commit
/// outcomes are wrapped as Request errors and must retain browser recovery data.
fn dispatch_error(error: ArangoError) -> HandlerError {
    match error {
        ArangoError::Api {
            status: 409,
            error_num: 1200,
            message,
            ..
        } => invalid(message),
        other => db_error(other),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Reference {
    pub thread_id: String,
    pub turn_id: String,
    pub item_id: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case", deny_unknown_fields)]
pub enum Request {
    Config,
    Threads {
        cursor: Option<String>,
    },
    History {
        thread_id: String,
        cursor: Option<String>,
    },
    Bindings {
        session: String,
    },
    Open {
        id: String,
        session: String,
        thread_id: Option<String>,
    },
    State {
        binding: String,
        before: Option<String>,
    },
    Send {
        binding: String,
        request_id: String,
        expected_revision: u64,
        message: String,
        highlights: Vec<String>,
        knowledge: Vec<String>,
        references: Vec<Reference>,
    },
    Interrupt {
        binding: String,
    },
    Release {
        binding: String,
    },
    Approval {
        binding: String,
        generation: String,
        request_id: Value,
        result: Value,
    },
    ReadPermission {
        binding: String,
        allowed: bool,
    },
    Reconcile {
        binding: String,
    },
    Position {
        binding: String,
        page: u32,
        block: Option<String>,
    },
}
struct Owned {
    worker: Arc<Worker>,
    approvals: Mutex<()>,
    thread: String,
    _capacity: SemaphorePermit<'static>,
}
static WORKERS: OnceLock<Mutex<BTreeMap<String, Arc<Owned>>>> = OnceLock::new();
static CAPACITY: Semaphore = Semaphore::const_new(4);
type BindingLocks = BTreeMap<String, Weak<Mutex<()>>>;
static BINDING_LOCKS: OnceLock<StdMutex<BindingLocks>> = OnceLock::new();
fn workers() -> &'static Mutex<BTreeMap<String, Arc<Owned>>> {
    WORKERS.get_or_init(|| Mutex::new(BTreeMap::new()))
}
/// Serialize open/send/release for one binding without blocking other bindings
/// or their polling, interruption, and approval requests. Idle locks are pruned.
async fn binding_lock(key: &str) -> OwnedMutexGuard<()> {
    let lock = {
        let mut locks = BINDING_LOCKS
            .get_or_init(|| StdMutex::new(BTreeMap::new()))
            .lock()
            .unwrap();
        locks.retain(|_, lock| lock.strong_count() > 0);
        locks.get(key).and_then(Weak::upgrade).unwrap_or_else(|| {
            let lock = Arc::new(Mutex::new(()));
            locks.insert(key.to_owned(), Arc::downgrade(&lock));
            lock
        })
    };
    lock.lock_owned().await
}
/// Clone ownership under the registry lock; never hold that lock across I/O.
async fn owner(key: &str) -> Option<Arc<Owned>> {
    workers().lock().await.get(key).cloned()
}
fn registry_key(pool: &ArangoPool, id: &str) -> String {
    format!(
        "{:?}/{:?}/{}/{}",
        pool.writer().base_url(),
        pool.writer().socket_path(),
        pool.database(),
        id
    )
}
async fn ensure_store(pool: &ArangoPool) -> Result<(), HandlerError> {
    for name in [BINDINGS, EXCHANGES] {
        match crud::create_collection(pool, name, Some(2)).await {
            Ok(_)
            | Err(ArangoError::Api {
                error_num: 1207, ..
            }) => {}
            Err(e) => return Err(db_error(e)),
        }
        let properties = pool
            .writer()
            .get(&format!("collection/{name}/properties"))
            .await
            .map_err(db_error)?;
        if properties["type"] != 2 {
            return Err(invalid("conversation store must use document collections"));
        }
    }
    Ok(())
}
async fn binding(pool: &ArangoPool, id: &str) -> Result<Value, HandlerError> {
    token(id).map_err(invalid)?;
    pool.writer()
        .get(&format!("document/{BINDINGS}/{id}"))
        .await
        .map_err(db_error)
}
async fn session(pool: &ArangoPool, id: &str) -> Result<Session, HandlerError> {
    token(id).map_err(invalid)?;
    let row = pool
        .writer()
        .get(&format!("document/{SESSIONS}/{id}"))
        .await
        .map_err(db_error)?;
    serde_json::from_value(row["state"].clone()).map_err(|e| failure(e.to_string()))
}
fn thread_id(id: &str) -> Result<(), HandlerError> {
    token(id).map_err(invalid)
}
async fn connect(config: &ChatConfig) -> Result<Arc<Worker>, HandlerError> {
    config.validate()?;
    Worker::start(config).await
}
fn terminal(state: &str) -> bool {
    matches!(state, "completed" | "failed" | "interrupted")
}

pub async fn exchange_context(pool: &ArangoPool, id: &str) -> Result<Value, HandlerError> {
    token(id).map_err(invalid)?;
    let row = pool
        .writer()
        .get(&format!("document/{EXCHANGES}/{id}"))
        .await
        .map_err(db_error)?;
    Ok(
        json!({"context":row["context"],"context_hash":row["context_hash"],"message":row["message"],"exchange_id":id}),
    )
}

pub async fn handle(
    pool: &ArangoPool,
    config: &ChatConfig,
    request: Request,
) -> Result<Value, HandlerError> {
    if matches!(request, Request::Config) {
        return Ok(json!({"enabled":config.enabled,"provider":"codex","version":"0.163.0"}));
    }
    if matches!(request, Request::Open { .. } | Request::Bindings { .. }) {
        ensure_store(pool).await?;
    }
    match request {
        Request::Config => unreachable!(),
        Request::Threads { cursor } => {
            let w = connect(config).await?;
            let result=w.call("thread/list",json!({"limit":30,"sourceKinds":["cli","vscode","exec","appServer"],"cursor":cursor})).await;
            let _ = w.stop().await;
            result
        }
        Request::History {
            thread_id: thread,
            cursor,
        } => {
            thread_id(&thread)?;
            let w = connect(config).await?;
            let result = w
                .call(
                    "thread/turns/list",
                    json!({"threadId":thread,"itemsView":"full","limit":10,"cursor":cursor}),
                )
                .await;
            let _ = w.stop().await;
            result
        }
        Request::Bindings { session: id } => {
            session(pool, &id).await?;
            let rows=rows(pool.writer(),"FOR b IN hades_reading_conversations FILTER b.session == @session SORT b.created_at DESC LIMIT 101 RETURN b",json!({"session":id})).await.map_err(db_error)?;
            Ok(
                json!({"bindings":rows.iter().take(100).collect::<Vec<_>>(),"partial":rows.len()>100}),
            )
        }
        Request::Open {
            id,
            session: session_id,
            thread_id: thread,
        } => open(pool, config, id, session_id, thread).await,
        Request::State {
            binding: id,
            before,
        } => {
            let row = binding(pool, &id).await?;
            let w = workers()
                .lock()
                .await
                .get(&registry_key(pool, &id))
                .map(|o| o.worker.clone());
            let live = if let Some(w) = w {
                w.live.read().await.value(&w.generation)
            } else {
                json!({"status":"disconnected","approvals":[],"items":[]})
            };
            let cursor = if let Some(before) = before {
                token(&before).map_err(invalid)?;
                let cursor = pool
                    .writer()
                    .get(&format!("document/{EXCHANGES}/{before}"))
                    .await
                    .map_err(db_error)?;
                if cursor["binding"] != id {
                    return Err(invalid("History cursor belongs to another binding"));
                }
                cursor
            } else {
                Value::Null
            };
            let records=rows(pool.writer(),"FOR e IN hades_reading_exchanges FILTER e.binding == @binding FILTER @date == null OR e.created_at < @date OR (e.created_at == @date AND e._key < @key) SORT e.created_at DESC, e._key DESC LIMIT 21 RETURN e",json!({"binding":id,"date":cursor["created_at"],"key":cursor["_key"]})).await.map_err(db_error)?;
            let next = if records.len() > 20 {
                records[19]["_key"].clone()
            } else {
                Value::Null
            };
            let records: Vec<_> = records.into_iter().take(20).rev().collect();
            // Transcript pages and browser caches may omit an exchange or retain
            // its old status. Controls need current state for the whole binding.
            let unresolved = rows(pool.writer(), "FOR e IN hades_reading_exchanges FILTER e.binding == @binding AND e.status NOT IN ['completed','failed','interrupted'] RETURN e.status", json!({"binding":id})).await.map_err(db_error)?;
            let pending = json!({"unresolved":!unresolved.is_empty(),"active":unresolved.iter().any(|status| matches!(status.as_str(), Some("dispatching" | "acknowledged")))});
            Ok(
                json!({"binding":row,"live":live,"exchanges":records,"next_before":next,"pending":pending}),
            )
        }
        Request::Send {
            binding: id,
            request_id,
            expected_revision,
            message,
            highlights,
            knowledge,
            references,
        } => match send(
            pool,
            config,
            SendInput {
                binding: id,
                request_id,
                expected_revision,
                message,
                highlights,
                knowledge,
                references,
            },
        )
        .await
        {
            Err(e @ (HandlerError::InvalidParameter { .. } | HandlerError::ServiceError(_))) => {
                Ok(json!({"rejected":true,"error":e.to_string()}))
            }
            other => other,
        },
        Request::Interrupt { binding: id } => {
            let owned = owner(&registry_key(pool, &id))
                .await
                .ok_or_else(|| failure("Conversation is not connected"))?;
            let turn = owned
                .worker
                .live
                .read()
                .await
                .turn
                .clone()
                .ok_or_else(|| failure("No active turn"))?;
            owned
                .worker
                .call(
                    "turn/interrupt",
                    json!({"threadId":owned.thread,"turnId":turn}),
                )
                .await
        }
        Request::Release { binding: id } => {
            let key = registry_key(pool, &id);
            let _binding_guard = binding_lock(&key).await;
            binding(pool, &id).await?;
            if let Some(owned) = owner(&key).await {
                let active = rows(pool.writer(), "FOR e IN hades_reading_exchanges FILTER e.binding == @binding AND e.status IN ['dispatching','acknowledged'] LIMIT 1 RETURN e._key", json!({"binding":id})).await.map_err(db_error)?;
                let live = owned.worker.live.read().await;
                if live.status == "inProgress" || (!live.closed && !active.is_empty()) {
                    return Err(failure(
                        "Stop the response and wait for completion before releasing this conversation",
                    ));
                }
                drop(live);
                owned.worker.stop().await?;
            }
            workers().lock().await.remove(&key);
            Ok(json!({"released":true}))
        }
        Request::ReadPermission {
            binding: id,
            allowed,
        } => {
            let key = registry_key(pool, &id);
            let _guard = binding_lock(&key).await;
            binding(pool, &id).await?;
            let owned = owner(&key).await;
            let persist = async {
                pool.writer()
                    .patch(
                        &format!("document/{BINDINGS}/{id}"),
                        &json!({"allow_reads":allowed}),
                    )
                    .await
                    .map_err(db_error)
            };
            if let Some(owned) = &owned {
                // Serialize manual decisions with transitions without placing
                // approvals behind a potentially slow open/send lifecycle lock.
                let _approval_guard = owned.approvals.lock().await;
                let mut live =
                    persist_read_permission(&owned.worker.live, allowed, persist).await?;
                owned.worker.approve_pending_reads(&mut live).await?;
            } else {
                persist.await?;
            }
            Ok(json!({"allow_reads":allowed}))
        }
        Request::Approval {
            binding: id,
            generation,
            request_id,
            result,
        } => {
            let owned = owner(&registry_key(pool, &id))
                .await
                .ok_or_else(|| failure("Approval connection has ended"))?;
            if generation != owned.worker.generation {
                return Err(failure("This approval belongs to an earlier connection"));
            }
            let _approval_guard = owned.approvals.lock().await;
            let mut live = owned.worker.live.write().await;
            let prompt = live
                .approvals
                .get(&request_id.to_string())
                .ok_or_else(|| failure("Approval is no longer pending"))?;
            validate_approval(prompt, &result)?;
            owned
                .worker
                .send(json!({"id":request_id,"result":result}))
                .await?;
            live.approvals.remove(&request_id.to_string());
            Ok(json!({"answered":true}))
        }
        Request::Reconcile { binding: id } => reconcile(pool, config, &id).await,
        Request::Position {
            binding: id,
            page,
            block,
        } => {
            binding(pool, &id).await?;
            if page == 0 || page > 100000 || block.as_ref().is_some_and(|b| b.len() > 256) {
                return Err(invalid("invalid reading position"));
            }
            pool.writer()
                .patch(
                    &format!("document/{BINDINGS}/{id}"),
                    &json!({"position":{"page":page,"block":block}}),
                )
                .await
                .map_err(db_error)?;
            Ok(json!({"saved":true}))
        }
    }
}

async fn open(
    pool: &ArangoPool,
    config: &ChatConfig,
    id: String,
    session_id: String,
    thread: Option<String>,
) -> Result<Value, HandlerError> {
    token(&id).map_err(invalid)?;
    if let Some(t) = &thread {
        thread_id(t)?;
    }
    config.validate()?;
    let key = registry_key(pool, &id);
    let _binding_guard = binding_lock(&key).await;
    let s = session(pool, &session_id).await?;
    let identity = config.identity();
    let existing = optional(pool.writer(), &format!("document/{BINDINGS}/{id}"))
        .await
        .map_err(db_error)?;
    let is_new = existing.is_none();
    let mut record = if let Some(mut row) = existing {
        if row["session"] != session_id
            || row["profile"] != identity
            || row["source_hash"] != s.source_hash
            || (thread.is_some()
                && !row["thread_id"].is_null()
                && row["thread_id"] != json!(thread))
        {
            return Err(invalid(
                "conversation binding belongs to a different session, profile, or thread",
            ));
        }
        if let Some(owned) = owner(&key).await {
            if !owned.worker.live.read().await.closed {
                return Ok(row);
            }
            owned.worker.stop().await?;
            workers().lock().await.remove(&key);
        }
        if row["thread_id"].is_null() {
            if thread.is_none() {
                return Err(failure(
                    "Conversation creation outcome is unknown; inspect Codex history and explicitly link a thread",
                ));
            }
            row["thread_id"] = json!(thread);
        }
        row
    } else {
        json!({"_key":id,"format":"hades-reading-chat/v1","session":session_id,"source_hash":s.source_hash,"profile":identity,"provider":"codex","thread_id":thread,"created_at":chrono::Utc::now().to_rfc3339(),"position":{"page":1,"block":null}})
    };
    // Reserve capacity before connecting or persisting a binding. Cancellation
    // and early failures release this permit automatically.
    let capacity = CAPACITY.try_acquire().map_err(|_| {
        failure("Release an existing conversation before opening another (four-worker limit)")
    })?;
    let w = connect(config).await?;
    let outcome=async {
        let policy=w.profile(config).await?;
        // Only thread/start can create an ambiguous provider outcome. Preparation
        // failures leave no new binding; once dispatch is possible, keep the row.
        if is_new {
            pool.writer().post(&format!("document/{BINDINGS}"), &record).await.map_err(db_error)?;
        }
        let mut params=json!({"cwd":config.cwd,"sandbox":"read-only","approvalPolicy":"on-request","approvalsReviewer":"user","config":policy,"developerInstructions":"You are discussing a document with its reader. The user message contains a frozen HADES exchange snapshot. Read reading_exchange_context with its exchange_id for that exact snapshot. Live reading_context may have changed; label any newer context. Use only the configured HADES guided-reading tools for retrieval and proposals. Proposals require an existing human highlight and remain pending until the human accepts in HADES. Never bypass this review through shell, direct database writes, or another endpoint."});
        let method=if record["thread_id"].is_string(){params["threadId"]=record["thread_id"].clone();params["excludeTurns"]=json!(true);"thread/resume"}else{"thread/start"};
        let response = w.call(method,params).await?;
        let thread = response["thread"]["id"]
        .as_str()
        .ok_or_else(|| failure("Codex returned no thread ID"))?
        .to_owned();
        w.live.write().await.thread = Some(thread.clone());
        record["thread_id"] = json!(thread);
        // Keep the known thread ID even if the following tool audit fails, so
        // correcting the profile can resume this thread instead of creating one.
        pool
        .writer()
        .patch(&format!("document/{BINDINGS}/{id}"), &json!({"thread_id":thread}))
        .await
        .map_err(db_error)?;
        w.verify_tools(&thread).await?;
        w.live.write().await.allow_reads = record["allow_reads"] == true;
        Ok(thread)
    }.await;
    let thread = match outcome {
        Ok(thread) => thread,
        Err(e) => {
            let _ = w.stop().await;
            return Err(e);
        }
    };
    workers().lock().await.insert(
        key,
        Arc::new(Owned {
            worker: w,
            approvals: Mutex::new(()),
            thread,
            _capacity: capacity,
        }),
    );
    Ok(record)
}

/// Keep the provider event stream and polling responsive during persistence.
/// Callers serialize permission transitions and manual approvals separately.
async fn persist_read_permission<'a>(
    live: &'a tokio::sync::RwLock<runtime::Live>,
    allowed: bool,
    persist: impl std::future::Future<Output = Result<Value, HandlerError>>,
) -> Result<tokio::sync::RwLockWriteGuard<'a, runtime::Live>, HandlerError> {
    live.write().await.allow_reads = false;
    persist.await?;
    let mut live = live.write().await;
    live.allow_reads = allowed;
    Ok(live)
}

#[derive(Clone, Serialize)]
struct SendInput {
    binding: String,
    request_id: String,
    expected_revision: u64,
    message: String,
    highlights: Vec<String>,
    knowledge: Vec<String>,
    references: Vec<Reference>,
}
fn snapshot(
    s: &Session,
    p: &SendInput,
    database: &str,
    refs: Vec<Value>,
) -> Result<Value, HandlerError> {
    if s.revision != p.expected_revision {
        return Err(invalid(
            "Reading changed; review the selected context before sending",
        ));
    }
    if p.highlights.len() > 6 || p.references.len() > 6 || p.knowledge.len() > 6 {
        return Err(invalid(
            "Select at most six passages, six accepted notes, and six conversation references",
        ));
    }
    let mut highlights = Vec::new();
    for id in &p.highlights {
        let h = s
            .highlights
            .get(id)
            .filter(|h| !h.removed)
            .ok_or_else(|| invalid("Selected highlight is unavailable"))?;
        highlights.push(json!({"id":id,"label":h.label,"anchor":h.anchor,"current_passage_revision":h.anchor.block.as_deref().map(|b|s.passage_revision(b))}));
    }
    let mut knowledge = Vec::new();
    for id in &p.knowledge {
        let proposal = s
            .proposals
            .get(id)
            .filter(|v| v.status == "accepted")
            .ok_or_else(|| invalid("Selected knowledge is not accepted in this reading"))?;
        knowledge.push(json!({"id":id,"revision":proposal.revision,"graph_ids":proposal.created_ids,"draft":proposal.draft}));
    }
    let value = json!({"format":"hades-reading-context/v1","database":database,"session":s.id,"source_hash":s.source_hash,"session_revision":s.revision,"title":s.title,"bound":s.bound,"highlights":highlights,"knowledge":knowledge,"conversation_references":refs});
    if serde_json::to_vec(&value).unwrap().len() > 12000 {
        return Err(invalid(
            "Selected context exceeds 12,000 bytes; select fewer passages or references",
        ));
    }
    Ok(value)
}
async fn send(
    pool: &ArangoPool,
    _config: &ChatConfig,
    p: SendInput,
) -> Result<Value, HandlerError> {
    token(&p.request_id).map_err(invalid)?;
    if p.message.trim().is_empty() || p.message.len() > 8000 {
        return Err(invalid("Message must contain 1–8,000 bytes"));
    }
    let b = binding(pool, &p.binding).await?;
    let id = record_hash(&json!([p.binding, p.request_id]));
    let digest = record_hash(&serde_json::to_value(&p).unwrap());
    if let Some(old) = optional(pool.writer(), &format!("document/{EXCHANGES}/{id}"))
        .await
        .map_err(db_error)?
    {
        if old["request_hash"] != digest {
            return Err(invalid(
                "Request ID was already used with a different message or context",
            ));
        }
        return Ok(old);
    }
    let key = registry_key(pool, &p.binding);
    let _binding_guard = binding_lock(&key).await;
    // Another browser can commit the same request while we wait for the owner.
    if let Some(old) = optional(pool.writer(), &format!("document/{EXCHANGES}/{id}"))
        .await
        .map_err(db_error)?
    {
        return if old["request_hash"] == digest {
            Ok(old)
        } else {
            Err(invalid(
                "Request ID was already used with a different message or context",
            ))
        };
    }
    let owned = owner(&key)
        .await
        .ok_or_else(|| failure("Resume this conversation before sending"))?;
    let live = owned.worker.live.read().await;
    if live.status == "inProgress" || live.closed {
        return Err(failure("Conversation is busy or disconnected"));
    }
    drop(live);
    let s = session(pool, b["session"].as_str().unwrap_or("")).await?;
    let mut references = Vec::new();
    if p.references.len() > 6 {
        return Err(invalid("Too many conversation references"));
    }
    for reference in &p.references {
        thread_id(&reference.thread_id)?;
        let history = owned.worker.history(&reference.thread_id).await?;
        let item = history
            .iter()
            .find(|t| t["id"] == reference.turn_id)
            .and_then(|t| t["items"].as_array())
            .and_then(|items| items.iter().find(|i| i["id"] == reference.item_id))
            .ok_or_else(|| invalid("Referenced conversation message is unavailable"))?;
        if !matches!(item["type"].as_str(), Some("userMessage" | "agentMessage")) {
            return Err(invalid(
                "Reference user or assistant messages, not tool output",
            ));
        }
        references.push(json!({"reference":reference,"item":item,"digest":record_hash(item),"captured_at":chrono::Utc::now().to_rfc3339()}));
    }
    let context = snapshot(&s, &p, pool.database(), references)?;
    let record = json!({"_key":id,"binding":p.binding,"request_id":p.request_id,"client_id":id,"request_hash":digest,"context_hash":record_hash(&context),"context":context,"message":p.message,"status":"dispatching","provider_turn":null,"projection":null,"created_at":chrono::Utc::now().to_rfc3339()});
    let copy = record.clone();
    let session_id = s.id.clone();
    let revision = s.revision;
    let record=transaction::run_with_reads(pool,vec![EXCHANGES.into()],vec![SESSIONS.into()],move|client|async move{
        if let Some(old)=optional(&client,&format!("document/{EXCHANGES}/{}",copy["_key"].as_str().unwrap())).await? {
            if old["request_hash"]!=copy["request_hash"]{return Err(conflict("request identity changed"));}return Ok((old,false));
        }
        let current=client.get(&format!("document/{SESSIONS}/{session_id}")).await?;
        if current["state"]["revision"]!=revision{return Err(conflict("Reading changed before dispatch; review context again"));}
        let previous=rows(&client,"FOR e IN hades_reading_exchanges FILTER e.binding == @binding RETURN {status:e.status}",json!({"binding":copy["binding"]})).await?;
        if previous.len()>=200{return Err(conflict("Conversation binding reached 200 exchanges; create a new binding"));}
        if previous.iter().any(|e|!terminal(e["status"].as_str().unwrap_or("unknown"))){return Err(conflict("An earlier exchange is unresolved; reconcile it before sending"));}
        client.post(&format!("document/{EXCHANGES}"),&copy).await?;Ok((copy,true))
    }).await.map_err(dispatch_error)?;
    if !record.1 {
        return Ok(record.0);
    }
    let w = owned.worker.clone();
    let thread = owned.thread.clone();
    let record = record.0;
    let pool = pool.clone();
    // Once journaled, a detached task owns dispatch even if the HTTP request vanishes.
    let result = record.clone();
    tokio::spawn(async move {
        let prompt=json!({"exchange_id":record["_key"],"context":record["context"],"message":record["message"]}).to_string();
        let started=w.call("turn/start",json!({"threadId":thread,"clientUserMessageId":record["client_id"],"input":[{"type":"text","text":prompt}]})).await;
        let id = record["_key"].as_str().unwrap();
        let turn = match started {
            Ok(v) => v["turn"]["id"].as_str().map(str::to_owned),
            Err(e) => {
                let _ =
                    update_unresolved(&pool, id, json!({"status":"unknown","error":e.to_string()}))
                        .await;
                return;
            }
        };
        let Some(turn) = turn else {
            let _ = update_unresolved(&pool, id, json!({"status":"unknown"})).await;
            return;
        };
        if matches!(
            update_unresolved(
                &pool,
                id,
                json!({"status":"acknowledged","provider_turn":turn})
            )
            .await,
            Ok(false)
        ) {
            return;
        }
        monitor_exchange(&pool, &w.live, &w.generation, id, &turn, 7200).await;
    });
    Ok(result)
}

/// A monitor must never replace a terminal result already saved by reconciliation.
/// The exclusive transaction makes the status check and update atomic with all
/// other writers, including reconciliation's ordinary document PATCH.
async fn update_unresolved(pool: &ArangoPool, id: &str, patch: Value) -> Result<bool, ArangoError> {
    let path = format!("document/{EXCHANGES}/{id}");
    transaction::run(pool, vec![EXCHANGES.into()], move |client| async move {
        let row = client.get(&path).await?;
        if terminal(row["status"].as_str().unwrap_or("")) {
            return Ok(false);
        }
        client.patch(&path, &patch).await?;
        Ok(true)
    })
    .await
}

/// Follow one turn until it is persisted or recovered, even after a newer turn
/// replaces the worker's live projection. Expiry only changes unresolved rows.
async fn monitor_exchange(
    pool: &ArangoPool,
    state: &tokio::sync::RwLock<runtime::Live>,
    generation: &str,
    id: &str,
    turn: &str,
    attempts: usize,
) {
    for _ in 0..attempts {
        tokio::time::sleep(Duration::from_millis(500)).await;
        if let Ok(row) = pool
            .writer()
            .get(&format!("document/{EXCHANGES}/{id}"))
            .await
            && terminal(row["status"].as_str().unwrap_or(""))
        {
            return;
        }
        let live = state.read().await;
        if live.turn.as_deref() == Some(turn)
            && (terminal(&live.status) || live.closed || live.status == "unknown")
        {
            let projection = live.value(generation);
            let status = live.status.clone();
            drop(live);
            if update_unresolved(
                pool,
                id,
                json!({"status":status,"projection":projection,"provider_turn":turn}),
            )
            .await
            .is_ok()
            {
                return;
            }
        }
    }
    let _ = update_unresolved(
        pool,
        id,
        json!({"status":"unknown","error":"Response deadline exceeded; reconcile history"}),
    )
    .await;
}

async fn reconcile(
    pool: &ArangoPool,
    config: &ChatConfig,
    id: &str,
) -> Result<Value, HandlerError> {
    let b = binding(pool, id).await?;
    let w = connect(config).await?;
    let history = w
        .history(
            b["thread_id"]
                .as_str()
                .ok_or_else(|| failure("Thread creation is unresolved"))?,
        )
        .await;
    let _ = w.stop().await;
    let history = history?;
    let records = rows(
        pool.writer(),
        "FOR e IN hades_reading_exchanges FILTER e.binding == @binding RETURN e",
        json!({"binding":id}),
    )
    .await
    .map_err(db_error)?;
    let mut unresolved = 0;
    for e in records {
        if terminal(e["status"].as_str().unwrap_or("")) {
            continue;
        }
        let matches: Vec<_> = history
            .iter()
            .filter(|t| {
                t["items"].as_array().is_some_and(|items| {
                    items
                        .iter()
                        .any(|i| i["type"] == "userMessage" && i["clientId"] == e["client_id"])
                })
            })
            .collect();
        if matches.len() == 1 && terminal(matches[0]["status"].as_str().unwrap_or("")) {
            let turn = matches[0];
            let projection = runtime::Live::from_turn(turn).value("");
            pool.writer().patch(&format!("document/{EXCHANGES}/{}",e["_key"].as_str().unwrap()),&json!({"provider_turn":turn["id"],"status":turn["status"],"projection":projection})).await.map_err(db_error)?;
        } else {
            unresolved += 1;
        }
    }
    Ok(json!({"unresolved":unresolved,"resent":false}))
}

fn validate_approval(prompt: &Value, result: &Value) -> Result<(), HandlerError> {
    match prompt["method"].as_str().unwrap_or("") {
        // Reading sessions cannot grant shell/filesystem escalation to bypass graph review.
        "item/commandExecution/requestApproval" | "item/fileChange/requestApproval" => {
            if !matches!(result["decision"].as_str(), Some("decline" | "cancel")) {
                return Err(invalid(
                    "Command and file permissions are not granted by the reading desk",
                ));
            }
        }
        "item/tool/requestUserInput" => {
            let answers = result["answers"]
                .as_object()
                .ok_or_else(|| invalid("Missing answers"))?;
            let questions = prompt["params"]["questions"]
                .as_array()
                .ok_or_else(|| invalid("Missing questions"))?;
            if answers.len() != questions.len() {
                return Err(invalid("Answer each pending question"));
            }
            for q in questions {
                let a = &result["answers"][q["id"].as_str().unwrap_or("")]["answers"];
                if !a.is_array()
                    || a.as_array().unwrap().len() != 1
                    || a[0].as_str().is_none_or(|s| s.len() > 4000)
                {
                    return Err(invalid("Invalid question answer"));
                }
            }
        }
        "mcpServer/elicitation/request" => {
            let allow_once = prompt["reading_tool"].is_string()
                && result == &json!({"action":"accept","content":{}});
            if !allow_once && !matches!(result["action"].as_str(), Some("decline" | "cancel")) {
                return Err(invalid(
                    "Only the pending allowlisted reading tool can be approved once",
                ));
            }
        }
        "item/permissions/requestApproval" => {
            if result["permissions"] != json!({}) {
                return Err(invalid(
                    "Additional permissions cannot be granted by the reading desk",
                ));
            }
        }
        _ => return Err(invalid("Unsupported provider request; stop the turn")),
    }
    if serde_json::to_vec(result).unwrap().len() > 16000 {
        return Err(invalid("Approval response too large"));
    }
    Ok(())
}

/// Release provider workers during orderly daemon shutdown.
pub async fn shutdown() {
    let registry = std::mem::take(&mut *workers().lock().await);
    for owned in registry.values() {
        let _ = owned.worker.stop().await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{Fixtures, with_temp_db};

    #[tokio::test]
    async fn monitor_preserves_recovery_after_live_turn_changes_and_at_expiry() {
        with_temp_db("chat_monitor", Fixtures::Empty, |pool| async move {
            ensure_store(&pool).await.unwrap();
            let live = tokio::sync::RwLock::new(runtime::Live {
                turn: Some("new_turn".into()),
                status: "inProgress".into(),
                ..Default::default()
            });
            for status in ["completed", "failed", "interrupted"] {
                let path = format!("document/{EXCHANGES}/{status}");
                pool.writer().post(&format!("document/{EXCHANGES}"), &json!({
                    "_key":status, "status":"acknowledged", "provider_turn":"old_turn"
                })).await.unwrap();
                // Same durable update as reconciliation, before the old monitor
                // sees completion and after a new turn replaces its projection.
                pool.writer().patch(&path, &json!({"status":status,"projection":{"items":[{"text":"Recovered answer"}]}})).await.unwrap();
                let recovered = pool.writer().get(&path).await.unwrap();
                tokio::time::timeout(Duration::from_secs(2), monitor_exchange(
                    &pool, &live, "generation", status, "old_turn", 7200,
                )).await.expect("monitor did not stop after journal recovery");
                // Also cover a reconciliation racing with each later write path.
                for patch in [json!({"status":"acknowledged"}), json!({"status":"unknown","error":"lost acknowledgement"}), json!({"status":"completed","projection":null})] {
                    assert!(!update_unresolved(&pool, status, patch).await.unwrap());
                }
                monitor_exchange(&pool, &live, "generation", status, "old_turn", 0).await;
                assert_eq!(pool.writer().get(&path).await.unwrap(), recovered, "terminal projection, status and revision must not change");
            }
            pool.writer().post(&format!("document/{EXCHANGES}"), &json!({"_key":"unresolved","status":"acknowledged"})).await.unwrap();
            monitor_exchange(&pool, &live, "generation", "unresolved", "old_turn", 0).await;
            let expired = pool.writer().get(&format!("document/{EXCHANGES}/unresolved")).await.unwrap();
            assert_eq!(expired["status"], "unknown");
            assert!(expired["error"].as_str().unwrap().contains("deadline exceeded"));
        }).await;
    }

    #[tokio::test]
    async fn monitor_update_serializes_with_reconciliation_commit() {
        with_temp_db("chat_monitor_race", Fixtures::Empty, |pool| async move {
            ensure_store(&pool).await.unwrap();
            let path = format!("document/{EXCHANGES}/race");
            pool.writer().post(&format!("document/{EXCHANGES}"), &json!({"_key":"race","status":"acknowledged"})).await.unwrap();
            let (ready, started) = tokio::sync::oneshot::channel();
            let (commit, committed) = tokio::sync::oneshot::channel();
            let copy = pool.clone();
            let recovery = tokio::spawn(async move {
                transaction::run(&copy, vec![EXCHANGES.into()], move |client| async move {
                    client.patch(&format!("document/{EXCHANGES}/race"), &json!({"status":"completed","projection":{"items":[{"text":"Recovered answer"}]}})).await?;
                    ready.send(()).unwrap();
                    committed.await.unwrap();
                    Ok(())
                }).await
            });
            started.await.unwrap();
            let update = update_unresolved(&pool, "race", json!({"status":"unknown","error":"expired"}));
            tokio::pin!(update);
            assert!(tokio::time::timeout(Duration::from_millis(50), &mut update).await.is_err());
            commit.send(()).unwrap();
            recovery.await.unwrap().unwrap();
            assert!(!update.await.unwrap(), "late update must observe the committed terminal state");
            let saved = pool.writer().get(&path).await.unwrap();
            assert_eq!(saved["status"], "completed");
            assert_eq!(saved["projection"]["items"][0]["text"], "Recovered answer");
            assert!(saved.get("error").is_none());
        }).await;
    }

    #[test]
    fn dispatch_conflicts_are_definitive_but_uncertain_commits_are_not() {
        assert!(matches!(
            dispatch_error(conflict("Reading changed before dispatch")),
            HandlerError::InvalidParameter { .. }
        ));
        assert!(matches!(
            dispatch_error(ArangoError::Request(
                "transaction commit outcome uncertain: connection lost".into()
            )),
            HandlerError::Query { .. }
        ));
    }
    #[tokio::test]
    async fn permission_persistence_keeps_provider_state_responsive_and_fails_closed() {
        for succeed in [false, true] {
            let live = Arc::new(tokio::sync::RwLock::new(runtime::Live::default()));
            live.write().await.allow_reads = true;
            let (started, ready) = tokio::sync::oneshot::channel();
            let (finish, finished) = tokio::sync::oneshot::channel();
            let pending_live = live.clone();
            let task = tokio::spawn(async move {
                persist_read_permission(&pending_live, true, async move {
                    started.send(()).unwrap();
                    finished.await.unwrap();
                    if succeed {
                        Ok(json!({}))
                    } else {
                        Err(failure("write failed"))
                    }
                })
                .await
                .is_ok()
            });
            ready.await.unwrap();
            // Provider events, polling and monitoring can acquire state while
            // the database write is deliberately held at the barrier.
            {
                let mut state = live
                    .try_write()
                    .expect("persistence held the live state lock");
                assert!(!state.allow_reads, "grant must wait for persistence");
                state.status = "completed".into();
            }
            finish.send(()).unwrap();
            assert_eq!(task.await.unwrap(), succeed);
            let state = live.read().await;
            assert_eq!(state.allow_reads, succeed);
            assert_eq!(state.status, "completed");
        }
    }
    #[test]
    fn permissions_cannot_bypass_reading_review() {
        for method in [
            "item/commandExecution/requestApproval",
            "item/fileChange/requestApproval",
        ] {
            let prompt = json!({"method":method});
            assert!(validate_approval(&prompt, &json!({"decision":"accept"})).is_err());
            assert!(validate_approval(&prompt, &json!({"decision":"decline"})).is_ok());
        }
        let mut prompt = json!({"method":"mcpServer/elicitation/request"});
        assert!(validate_approval(&prompt, &json!({"action":"accept","content":{}})).is_err());
        prompt["reading_tool"] = json!("reading_propose");
        assert!(validate_approval(&prompt, &json!({"action":"accept","content":{}})).is_ok());
        assert!(
            validate_approval(
                &prompt,
                &json!({"action":"accept","content":{},"_meta":{"persist":"always"}})
            )
            .is_err()
        );
        assert!(
            validate_approval(
                &json!({"method":"item/permissions/requestApproval"}),
                &json!({"permissions":{"network":{"enabled":true}}})
            )
            .is_err()
        );
    }
    #[test]
    fn chat_control_is_admin_but_exchange_snapshot_is_agent_readable() {
        use crate::dispatch::{AccessTier, DaemonCommand};
        assert_eq!(
            DaemonCommand::ReadingChat(Request::Config).access_tier(),
            AccessTier::Admin
        );
        assert_eq!(
            DaemonCommand::ReadingExchangeContext {
                exchange_id: "e".into()
            }
            .access_tier(),
            AccessTier::Agent
        );
    }
}
