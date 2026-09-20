# LSP document state after failed opens

At `d9b0aa2`, `open_file` inserts the URI into its open set before awaiting file
I/O and the didOpen send. It removes the entry only when the send returns an
error. A read failure or caller cancellation can therefore leave a URI marked
open even though no notification reached the server. Concurrent opens can return
before the initial send, and close has no shared per-document operation ordering.

A private real LspSession with a synthetic JSON-RPC peer completed initialize and
readiness. Reading a one-byte invalid UTF-8 file failed; after replacing it with
valid text, document_symbols returned an empty result because the session skipped
didOpen while the peer still had no open document. The bounded test passed in
0.01 seconds and shut down its peer normally. This is not a production retrieval
quality measurement. Historical probe and selected hashes are retained in
`repros/lsp_document_state.rs` and `lsp-document-state-result.json`.

Remediation must publish open state only after a complete notification send,
leave retries possible after read failure/cancellation, and order competing
operations for the same URI without serializing unrelated files. Cancellation
must not leave permanent lock/state entries; close/reopen and normal symbol
requests need maintained private-peer tests.
