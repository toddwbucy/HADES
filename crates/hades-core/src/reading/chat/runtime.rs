//! Codex stdio owner. Browser disconnects never own this process's lifetime.
use super::{ChatConfig, failure};
use crate::dispatch::HandlerError;
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    process::Stdio,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    process::{Child, ChildStdin, Command},
    sync::{Mutex, RwLock, oneshot},
};

type Reply = oneshot::Sender<Result<Value, String>>;
const OUTPUT_LIMIT: usize = 256 * 1024;
const READING_TOOLS: &[&str] = &[
    "reading_context",
    "reading_exchange_context",
    "reading_propose",
    "db_query",
    "db_get",
    "db_lookup",
    "graph_neighbors",
];
const READ_TOOLS: &[&str] = &[
    "reading_context",
    "reading_exchange_context",
    "db_query",
    "db_get",
    "db_lookup",
    "graph_neighbors",
];

#[derive(Default)]
pub struct Live {
    pub allow_reads: bool,
    pub thread: Option<String>,
    pub order: Vec<String>,
    pub turn: Option<String>,
    pub status: String,
    pub items: BTreeMap<String, Value>,
    pub approvals: BTreeMap<String, Value>,
    pub error: Option<String>,
    pub truncated: bool,
    pub closed: bool,
}
impl Live {
    fn approval(&self, mut request: Value) -> Value {
        if let Some(object) = request.as_object_mut() {
            object.remove("reading_tool");
            object.remove("read_only");
        }
        let p = &request["params"];
        if request["method"] == "mcpServer/elicitation/request"
            && p["serverName"] == "hades_reading"
            && p["mode"] == "form"
            && p["_meta"]["codex_approval_kind"] == "mcp_tool_call"
            && p["requestedSchema"] == json!({"type":"object","properties":{}})
            && let Some(item) = self.items.values().find(|i| {
                i["type"] == "mcpToolCall"
                    && i["status"] == "inProgress"
                    && i["server"] == "hades_reading"
                    && i["arguments"] == p["_meta"]["tool_params"]
                    && i["tool"]
                        .as_str()
                        .is_some_and(|tool| READING_TOOLS.contains(&tool))
            })
        {
            // Do not infer a read grant from ambiguous concurrent calls with
            // identical arguments (a proposal could otherwise match a read).
            let matches = self
                .items
                .values()
                .filter(|i| {
                    i["type"] == "mcpToolCall"
                        && i["status"] == "inProgress"
                        && i["server"] == "hades_reading"
                        && i["arguments"] == p["_meta"]["tool_params"]
                })
                .count();
            if matches == 1 {
                request["reading_tool"] = item["tool"].clone();
                request["read_only"] = json!(READ_TOOLS.contains(&item["tool"].as_str().unwrap()));
            }
        }
        request
    }
    fn auto_approve(&self, request: &Value) -> bool {
        self.allow_reads
            && !self.closed
            && self.status == "inProgress"
            && request["params"]["threadId"].as_str() == self.thread.as_deref()
            && request["params"]["turnId"].as_str() == self.turn.as_deref()
            && self.approval(request.clone())["read_only"] == true
    }
    pub fn from_turn(turn: &Value) -> Self {
        let mut state = Self::default();
        state.event(&json!({"method":"turn/started","params":{"turn":turn}}));
        for item in turn["items"].as_array().into_iter().flatten() {
            state.event(
                &json!({"method":"item/completed","params":{"turnId":turn["id"],"item":item}}),
            );
        }
        state.event(&json!({"method":"turn/completed","params":{"turn":turn}}));
        state
    }
    pub fn value(&self, generation: &str) -> Value {
        json!({"turn_id":self.turn,"status":self.status,"items":self.order.iter().filter_map(|id|self.items.get(id)).collect::<Vec<_>>(),"approvals":self.approvals.values().collect::<Vec<_>>(),"error":self.error,"truncated":self.truncated,"closed":self.closed,"generation":generation})
    }
    fn event(&mut self, event: &Value) {
        let p = &event["params"];
        let method = event["method"].as_str().unwrap_or("");
        if let Some(thread) = p["threadId"].as_str() {
            if self
                .thread
                .as_deref()
                .is_some_and(|expected| expected != thread)
            {
                return;
            }
            self.thread = Some(thread.into());
        }
        if method == "turn/started" {
            self.turn = p["turn"]["id"].as_str().map(str::to_owned);
            self.status = "inProgress".into();
            self.items.clear();
            self.order.clear();
            self.approvals.clear();
            self.error = None;
            self.truncated = false;
        }
        if let Some(turn) = p["turnId"].as_str()
            && self.turn.as_deref() != Some(turn)
        {
            return;
        }
        match method {
            "item/started" | "item/completed" => {
                if let Some(id) = p["item"]["id"].as_str() {
                    if !self.items.contains_key(id) {
                        self.order.push(id.into());
                    }
                    self.items.insert(id.into(), p["item"].clone());
                }
            }
            "item/agentMessage/delta" => {
                if let (Some(id), Some(delta)) = (p["itemId"].as_str(), p["delta"].as_str()) {
                    if !self.items.contains_key(id) {
                        self.order.push(id.into());
                    }
                    let item = self
                        .items
                        .entry(id.into())
                        .or_insert_with(|| json!({"id":id,"type":"agentMessage","text":""}));
                    let mut text = item["text"].as_str().unwrap_or("").to_owned();
                    text.push_str(delta);
                    item["text"] = json!(text);
                }
            }
            "turn/completed" if p["turn"]["id"].as_str() == self.turn.as_deref() => {
                self.status = p["turn"]["status"].as_str().unwrap_or("failed").into();
                self.error = p["turn"]["error"]["message"].as_str().map(str::to_owned);
                self.approvals.clear();
            }
            "serverRequest/resolved" => {
                self.approvals.remove(&p["requestId"].to_string());
            }
            _ => {}
        }
        // Keep a bounded UI projection. Full provider history remains in Codex.
        if serde_json::to_vec(&self.items).map_or(true, |v| v.len() > OUTPUT_LIMIT) {
            self.truncated = true;
            for item in self.items.values_mut() {
                if let Some(text) = item["text"].as_str() {
                    item["text"] = json!(text.chars().take(4000).collect::<String>());
                }
                if let Some(text) = item["aggregatedOutput"].as_str() {
                    item["aggregatedOutput"] = json!(text.chars().take(2000).collect::<String>());
                }
            }
            while !self.order.is_empty()
                && (self.items.len() > 32
                    || serde_json::to_vec(&self.items).map_or(true, |v| v.len() > OUTPUT_LIMIT))
            {
                let first = self.order.remove(0);
                self.items.remove(&first);
            }
        }
    }
}

pub struct Worker {
    pub live: Arc<RwLock<Live>>,
    pub generation: String,
    input: Arc<Mutex<ChildStdin>>,
    child: Mutex<Child>,
    pending: Arc<Mutex<BTreeMap<u64, Reply>>>,
    next: AtomicU64,
}
impl Worker {
    pub async fn start(config: &ChatConfig) -> Result<Arc<Self>, HandlerError> {
        let version = tokio::time::timeout(
            Duration::from_secs(10),
            Command::new(&config.executable).arg("--version").output(),
        )
        .await
        .map_err(|_| failure("Codex version check timed out"))?
        .map_err(|e| failure(e.to_string()))?;
        if String::from_utf8_lossy(&version.stdout).trim() != "codex-cli 0.163.0" {
            return Err(failure(
                "Codex reading currently requires verified codex-cli 0.163.0",
            ));
        }
        let mut command = Command::new(&config.executable);
        #[cfg(target_os = "linux")]
        {
            let parent = std::process::id() as libc::pid_t;
            // SAFETY: only async-signal-safe libc calls run between fork and exec.
            unsafe {
                command.pre_exec(move || {
                    if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) != 0 {
                        return Err(std::io::Error::last_os_error());
                    }
                    if libc::getppid() != parent {
                        return Err(std::io::Error::other("reading daemon exited"));
                    }
                    Ok(())
                });
            }
        }
        command
            .arg("app-server")
            .current_dir(&config.cwd)
            .env("CODEX_HOME", &config.home)
            .env_remove("ARANGO_PASSWORD")
            .env_remove("ARANGO_USERNAME")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true);
        for key in [
            "shell_tool",
            "hooks",
            "apps",
            "browser_use",
            "computer_use",
            "remote_plugin",
            "skill_mcp_dependency_install",
        ] {
            command.args(["-c", &format!("features.{key}=false")]);
        }
        command.args([
            "-c",
            "web_search=\"disabled\"",
            "-c",
            "approval_policy=\"on-request\"",
            "-c",
            "approvals_reviewer=\"user\"",
        ]);
        let mut child = command
            .spawn()
            .map_err(|e| failure(format!("Cannot start Codex: {e}")))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| failure("Missing Codex stdout"))?;
        let input = child
            .stdin
            .take()
            .ok_or_else(|| failure("Missing Codex stdin"))?;
        let live = Arc::new(RwLock::new(Live::default()));
        let input = Arc::new(Mutex::new(input));
        let pending: Arc<Mutex<BTreeMap<u64, Reply>>> = Arc::new(Mutex::new(BTreeMap::new()));
        let worker = Arc::new(Self {
            live: live.clone(),
            generation: format!("{:032x}", rand::random::<u128>()),
            input: input.clone(),
            child: Mutex::new(child),
            pending: pending.clone(),
            next: AtomicU64::new(1),
        });
        tokio::spawn(async move {
            let mut reader = BufReader::new(stdout);
            loop {
                let mut bytes = Vec::new();
                // read_until is limited independently of provider honesty.
                let result = tokio::io::AsyncReadExt::take(&mut reader, 2 * 1024 * 1024 + 1)
                    .read_until(b'\n', &mut bytes)
                    .await;
                if !matches!(result,Ok(n) if n>0 && n<=2*1024*1024) {
                    break;
                }
                let Ok(message) = serde_json::from_slice::<Value>(&bytes) else {
                    break;
                };
                if message.get("method").is_none() {
                    if let Some(id) = message["id"].as_u64()
                        && let Some(sender) = pending.lock().await.remove(&id)
                    {
                        let response = if message.get("error").is_some() {
                            Err(message["error"]["message"]
                                .as_str()
                                .unwrap_or("Codex RPC failed")
                                .into())
                        } else {
                            Ok(message["result"].clone())
                        };
                        let _ = sender.send(response);
                    }
                } else {
                    let mut state = live.write().await;
                    if message.get("id").is_some() {
                        let scoped = message["params"]["threadId"].as_str()
                            == state.thread.as_deref()
                            && message["params"]["turnId"].as_str() == state.turn.as_deref();
                        if !scoped || state.status != "inProgress" {
                            continue;
                        }
                        if state.approvals.len() < 8 && bytes.len() <= 16000 {
                            let request = state.approval(message);
                            if state.auto_approve(&request) {
                                if let Err(error) = send_to(&input, json!({"id":request["id"],"result":{"action":"accept","content":{}}})).await {
                                    state.error = Some(error.to_string());
                                    break;
                                }
                                continue;
                            }
                            state.approvals.insert(request["id"].to_string(), request);
                        } else {
                            state.error = Some(
                                "Codex request exceeds display limits; stop the response".into(),
                            );
                        }
                    } else {
                        state.event(&message);
                    }
                }
            }
            let mut state = live.write().await;
            state.closed = true;
            state.approvals.clear();
            if state.status == "inProgress" {
                state.status = "unknown".into();
                state.error =
                    Some("Codex disconnected; reconcile history before sending again".into());
            }
            for (_, sender) in std::mem::take(&mut *pending.lock().await) {
                let _ = sender.send(Err("Codex disconnected".into()));
            }
        });
        worker
            .call(
                "initialize",
                json!({"clientInfo":{"name":"hades_reading","version":env!("CARGO_PKG_VERSION")}}),
            )
            .await?;
        worker.send(json!({"method":"initialized"})).await?;
        Ok(worker)
    }
    pub async fn send(&self, value: Value) -> Result<(), HandlerError> {
        send_to(&self.input, value).await
    }
    pub async fn approve_pending_reads(&self, live: &mut Live) -> Result<(), HandlerError> {
        let ids: Vec<_> = live
            .approvals
            .iter()
            .filter(|(_, request)| live.auto_approve(request))
            .map(|(id, _)| id.clone())
            .collect();
        for id in ids {
            let request = &live.approvals[&id];
            self.send(json!({"id":request["id"],"result":{"action":"accept","content":{}}}))
                .await?;
            live.approvals.remove(&id);
        }
        Ok(())
    }
    pub async fn call(&self, method: &str, params: Value) -> Result<Value, HandlerError> {
        let id = self.next.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();
        self.pending.lock().await.insert(id, tx);
        let result = async {
            self.send(json!({"id":id,"method":method,"params":params}))
                .await?;
            tokio::time::timeout(Duration::from_secs(25), rx)
                .await
                .map_err(|_| failure("Codex request timed out; outcome may be unknown"))?
                .map_err(|_| failure("Codex disconnected"))?
                .map_err(failure)
        }
        .await;
        self.pending.lock().await.remove(&id);
        result
    }
    pub async fn stop(&self) -> Result<(), HandlerError> {
        let mut child = self.child.lock().await;
        if child
            .try_wait()
            .map_err(|e| failure(e.to_string()))?
            .is_none()
        {
            child.kill().await.map_err(|e| failure(e.to_string()))?;
        }
        child.wait().await.map_err(|e| failure(e.to_string()))?;
        Ok(())
    }
    pub async fn profile(&self, config: &ChatConfig) -> Result<Value, HandlerError> {
        let effective = self
            .call(
                "config/read",
                json!({"includeLayers":false,"cwd":config.cwd}),
            )
            .await?;
        let mut servers = serde_json::Map::new();
        if let Some(existing) = effective["config"]["mcp_servers"].as_object() {
            for key in existing.keys() {
                servers.insert(key.clone(), json!({"enabled":false}));
            }
        }
        servers.insert("hades_reading".into(),json!({"url":config.mcp_url,"bearer_token_env_var":config.mcp_token_env,"enabled":true,"required":true,"enabled_tools":READING_TOOLS}));
        let mut plugins = serde_json::Map::new();
        if let Some(existing) = effective["config"]["plugins"].as_object() {
            for key in existing.keys() {
                plugins.insert(key.clone(), json!({"enabled":false}));
            }
        }
        Ok(
            json!({"mcp_servers":servers,"plugins":plugins,"features":{"shell_tool":false,"hooks":false,"apps":false,"browser_use":false,"computer_use":false,"remote_plugin":false,"skill_mcp_dependency_install":false},"web_search":"disabled"}),
        )
    }
    pub async fn verify_tools(&self, thread: &str) -> Result<(), HandlerError> {
        let mut cursor = Value::Null;
        let mut found = false;
        for _ in 0..10 {
            let page = self
                .call(
                    "mcpServerStatus/list",
                    json!({"threadId":thread,"detail":"full","limit":100,"cursor":cursor}),
                )
                .await?;
            for server in page["data"]
                .as_array()
                .ok_or_else(|| failure("Invalid Codex MCP inventory"))?
            {
                if server["runtimeStatus"] == "disabled" {
                    continue;
                }
                if server["name"] != "hades_reading" || server["runtimeStatus"] != "connected" {
                    return Err(failure(
                        "Codex did not confirm the restricted guided-reading MCP connection",
                    ));
                }
                let tools = server["tools"]
                    .as_object()
                    .ok_or_else(|| failure("Missing guided-reading tools"))?;
                if tools
                    .keys()
                    .any(|name| !READING_TOOLS.contains(&name.as_str()))
                    || !READING_TOOLS.iter().all(|name| tools.contains_key(*name))
                {
                    return Err(failure(
                        "Codex MCP tool inventory differs from the reading allowlist",
                    ));
                }
                found = true;
            }
            cursor = page["nextCursor"].clone();
            if cursor.is_null() {
                return if found {
                    Ok(())
                } else {
                    Err(failure("Guided-reading MCP connection is missing"))
                };
            }
        }
        Err(failure("Codex MCP inventory pagination is incomplete"))
    }
    pub async fn history(&self, thread: &str) -> Result<Vec<Value>, HandlerError> {
        let mut cursor = Value::Null;
        let mut turns = Vec::new();
        let mut size = 0;
        for _ in 0..100 {
            let page=self.call("thread/turns/list",json!({"threadId":thread,"itemsView":"full","sortDirection":"asc","limit":20,"cursor":cursor})).await?;
            let data = page["data"]
                .as_array()
                .ok_or_else(|| failure("Invalid Codex history"))?;
            size += serde_json::to_vec(data).map_or(usize::MAX, |v| v.len());
            if size > 8 * 1024 * 1024 {
                return Err(failure(
                    "Conversation history exceeds recovery limit; outcome remains unknown",
                ));
            }
            turns.extend(data.iter().cloned());
            cursor = page["nextCursor"].clone();
            if cursor.is_null() {
                return Ok(turns);
            }
        }
        Err(failure(
            "History pagination incomplete; outcome remains unknown",
        ))
    }
}

async fn send_to(input: &Mutex<ChildStdin>, value: Value) -> Result<(), HandlerError> {
    let mut bytes = serde_json::to_vec(&value).map_err(|e| failure(e.to_string()))?;
    bytes.push(b'\n');
    input
        .lock()
        .await
        .write_all(&bytes)
        .await
        .map_err(|e| failure(format!("Codex connection lost: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stream_filters_foreign_threads_preserves_order_and_bounds_recovery() {
        let mut s = Live::default();
        s.event(
            &json!({"method":"turn/started","params":{"threadId":"ours","turn":{"id":"turn"}}}),
        );
        s.event(
            &json!({"method":"turn/started","params":{"threadId":"other","turn":{"id":"foreign"}}}),
        );
        assert_eq!(s.turn.as_deref(), Some("turn"));
        for id in ["z", "a"] {
            s.event(&json!({"method":"item/completed","params":{"threadId":"ours","turnId":"turn","item":{"id":id,"type":"agentMessage","text":"hello"}}}));
        }
        assert_eq!(s.value("")["items"][0]["id"], "z");
        let turn = json!({"id":"turn","status":"completed","items":[{"id":"huge","type":"mcpToolCall","result":"x".repeat(OUTPUT_LIMIT+1)}]});
        let recovered = Live::from_turn(&turn).value("");
        assert_eq!(recovered["truncated"], true);
        assert!(serde_json::to_vec(&recovered).unwrap().len() < OUTPUT_LIMIT);
    }
    #[test]
    fn reading_approval_requires_matching_live_tool_and_empty_form() {
        let mut s = Live::default();
        s.items.insert("tool".into(), json!({"type":"mcpToolCall","server":"hades_reading","tool":"reading_propose","status":"inProgress","arguments":{"session":"s"}}));
        let request = json!({"method":"mcpServer/elicitation/request","params":{"serverName":"hades_reading","mode":"form","requestedSchema":{"type":"object","properties":{}},"_meta":{"codex_approval_kind":"mcp_tool_call","tool_params":{"session":"s"}}}});
        assert_eq!(
            s.approval(request.clone())["reading_tool"],
            "reading_propose"
        );
        for field in ["serverName", "mode", "requestedSchema", "_meta"] {
            let mut changed = request.clone();
            changed["params"][field] = Value::Null;
            assert!(s.approval(changed)["reading_tool"].is_null());
        }
        s.items.get_mut("tool").unwrap()["tool"] = json!("kg_add_node");
        assert!(s.approval(request)["reading_tool"].is_null());
    }
    #[test]
    fn remembered_reads_require_current_unambiguous_read_call() {
        let mut live = Live {
            thread: Some("thread".into()),
            turn: Some("turn".into()),
            status: "inProgress".into(),
            ..Live::default()
        };
        let request = json!({"id":901,"method":"mcpServer/elicitation/request","params":{
            "threadId":"thread","turnId":"turn","serverName":"hades_reading","mode":"form",
            "requestedSchema":{"type":"object","properties":{}},
            "_meta":{"codex_approval_kind":"mcp_tool_call","tool_params":{"id":"one"}}}});
        live.items.insert(
            "call".into(),
            json!({"type":"mcpToolCall","server":"hades_reading",
            "tool":"db_get","status":"inProgress","arguments":{"id":"one"}}),
        );
        assert!(!live.auto_approve(&request));
        live.allow_reads = true;
        for tool in READ_TOOLS {
            live.items.get_mut("call").unwrap()["tool"] = json!(tool);
            assert!(live.auto_approve(&request), "{tool}");
        }
        for tool in ["reading_propose", "kg_add_node", "unknown"] {
            live.items.get_mut("call").unwrap()["tool"] = json!(tool);
            assert!(!live.auto_approve(&request));
        }
        live.items.get_mut("call").unwrap()["tool"] = json!("db_get");
        for field in [
            "threadId",
            "turnId",
            "serverName",
            "mode",
            "requestedSchema",
            "_meta",
        ] {
            let mut changed = request.clone();
            changed["params"][field] = Value::Null;
            changed["read_only"] = json!(true);
            changed["reading_tool"] = json!("db_get");
            assert!(!live.auto_approve(&changed), "{field}");
        }
        let mut other = live.items["call"].clone();
        other["tool"] = json!("reading_propose");
        live.items.insert("other".into(), other);
        assert!(!live.auto_approve(&request));
        live.items.remove("other");
        live.closed = true;
        assert!(!live.auto_approve(&request));
        live.closed = false;
        live.allow_reads = false;
        assert!(!live.auto_approve(&request));
    }
    #[test]
    fn terminal_event_clears_approvals_and_ignores_old_turns() {
        let mut s = Live::default();
        s.event(&json!({"method":"turn/started","params":{"turn":{"id":"new"}}}));
        s.approvals.insert("1".into(), json!({"id":1}));
        s.event(
            &json!({"method":"turn/completed","params":{"turn":{"id":"old","status":"completed"}}}),
        );
        assert_eq!(s.status, "inProgress");
        s.event(&json!({"method":"turn/completed","params":{"turn":{"id":"new","status":"interrupted"}}}));
        assert!(s.approvals.is_empty());
        assert_eq!(s.status, "interrupted");
        s.event(&json!({"method":"serverRequest/resolved","params":{"requestId":1}}));
        assert!(s.approvals.is_empty());
    }
}
