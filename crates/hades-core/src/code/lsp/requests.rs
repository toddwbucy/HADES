//! File-scoped semantic requests: a failure gets one deferred retry (#179).
use serde_json::Value;
use tracing::warn;

use super::{
    LspError,
    session::{LanguageServer, LspSession},
    symbols::{FAILED_REQUEST_LIMIT, FailedRequest, FileExtraction},
};

pub(super) enum RequestKind {
    Calls,
    Hover,
    Implementations { interface: String },
}

pub(super) struct SymbolRequest {
    pub index: usize,
    pub line: u32,
    pub character: u32,
    pub kind: RequestKind,
}

impl SymbolRequest {
    fn method(&self) -> &'static str {
        match self.kind {
            RequestKind::Calls => "callHierarchy/outgoingCalls",
            RequestKind::Hover => "textDocument/hover",
            RequestKind::Implementations { .. } => "textDocument/implementation",
        }
    }

    async fn execute<S: LanguageServer>(
        &self,
        session: &LspSession<S>,
        uri: &str,
    ) -> Result<Value, LspError> {
        match self.kind {
            RequestKind::Calls => session
                .call_hierarchy_outgoing(uri, self.line, self.character)
                .await
                .map(Value::Array),
            RequestKind::Hover => session
                .hover(uri, self.line, self.character)
                .await
                .map(|value| value.unwrap_or(Value::Null)),
            RequestKind::Implementations { .. } => session
                .implementations(uri, self.line, self.character)
                .await
                .map(Value::Array),
        }
    }
}

pub(super) async fn resolve<S: LanguageServer>(
    session: &LspSession<S>,
    uri: &str,
    file: &str,
    requests: Vec<SymbolRequest>,
    extraction: &mut FileExtraction,
) -> Vec<(SymbolRequest, Value)> {
    let mut complete = Vec::new();
    let mut retry = Vec::new();
    for request in requests {
        match request.execute(session, uri).await {
            Ok(value) => complete.push((request, value)),
            Err(_) => retry.push(request),
        }
    }
    // Do not immediately repeat an unready request: finish the file's other
    // requests first, giving the analyzer a chance to finish its work.
    for request in retry {
        match request.execute(session, uri).await {
            Ok(value) => complete.push((request, value)),
            Err(error) => {
                let symbol = &extraction.symbols[request.index].qualified_name;
                warn!(file, symbol, request = request.method(), %error, "semantic request failed after retry; existing file graph must be retained");
                extraction.failed_request_count += 1;
                if extraction.failed_requests.len() < FAILED_REQUEST_LIMIT {
                    extraction.failed_requests.push(FailedRequest {
                        file: file.chars().take(4096).collect(),
                        symbol: symbol.chars().take(1024).collect(),
                        request: request.method().into(),
                        reason: error.to_string().chars().take(2048).collect(),
                    });
                }
            }
        }
    }
    complete
}
