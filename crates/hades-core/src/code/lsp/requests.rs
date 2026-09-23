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
    pub range: Value,
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
            Err(error) => retry.push((request, error)),
        }
    }
    // Do not immediately repeat an unready request: finish the file's other
    // requests first, giving the analyzer a chance to finish its work.
    for (request, first_error) in retry {
        match request.execute(session, uri).await {
            Ok(value) => complete.push((request, value)),
            Err(error) => {
                let extracted = &extraction.symbols[request.index];
                let symbol = &extracted.qualified_name;
                let method = match &error {
                    LspError::Request { method, .. } => *method,
                    _ => request.method(),
                };
                let empty_prepare = matches!(&error, LspError::Request { method: "textDocument/prepareCallHierarchy", source }
                    if matches!(source.as_ref(), LspError::InvalidResponse(reason) if reason == "empty prepareCallHierarchy for callable symbol"));
                if empty_prepare && session.cfg_inactive(uri, &request.range).await {
                    extraction.no_call_count += 1;
                    if extraction.no_calls.len() < FAILED_REQUEST_LIMIT {
                        extraction.no_calls.push(FailedRequest {
                            file: file.chars().take(4096).collect(),
                            symbol: symbol.chars().take(1024).collect(),
                            request: method.into(),
                            reason: "cfg_inactive".into(),
                        });
                    }
                    complete.push((request, Value::Array(Vec::new())));
                    continue;
                }
                match &request.kind {
                    RequestKind::Calls => {
                        extraction.failed_edge_symbols.insert(symbol.clone());
                    }
                    RequestKind::Implementations { interface } => {
                        extraction.failed_edge_symbols.insert(interface.clone());
                    }
                    RequestKind::Hover => {}
                }
                warn!(file, symbol, request = method, %first_error, %error, "semantic request failed after retry; content remains eligible for refresh");
                extraction.failed_request_count += 1;
                if extraction.failed_requests.len() < FAILED_REQUEST_LIMIT {
                    extraction.failed_requests.push(FailedRequest {
                        file: file.chars().take(4096).collect(),
                        symbol: symbol.chars().take(1024).collect(),
                        request: method.into(),
                        reason: format!("first attempt: {first_error}; retry: {error}")
                            .chars()
                            .take(2048)
                            .collect(),
                    });
                }
            }
        }
    }
    complete
}
