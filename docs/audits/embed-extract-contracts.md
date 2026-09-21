# Embedding and extraction command contracts

Source cut: `92584154b1272072e81d8cc3946e583aea114a2f`. Seven leaves were traced
through declarations, main, CLI adapters and relevant embedding-client paths.
The [manifest](embed-extract-contracts.json) binds five sources. No services,
GPUs, converters or installation commands were invoked for this review.

## Native leaf contracts

Main loads configuration and applies overrides before routing, including for
file extraction and GPU listing. These adapters do not construct an ArangoDB
pool. Embedding uses the configured endpoint with HTTP transport, despite an
outdated module comment describing gRPC. Other controls use subprocesses.

| Leaf | Inputs/defaults | Result/error behavior |
|---|---|---|
| embed text | Required text; format json by default. Direct client sends one retrieval.passage input. | raw prints just the JSON vector; other formats use the shared formatter (json/jsonl/table, wider than help's json/raw wording). Full vector plus ten-value preview, dimension/model/duration and text preview (100 characters plus ellipsis). Format validation occurs after embedding. |
| embed service status | No local options; configured endpoint. | Fixed JSON, with ok/idle from model_loaded, error for info failure, or stopped for client-construction failure. Every branch returns successful outer status and zero exit. These states describe different failure stages, not necessarily process state. |
| embed service start | foreground=false; true is rejected. | Runs systemctl start hades-embedder, rejects spawn/nonzero failure, sleeps two seconds, then probes configured endpoint. Reports started and available even when available=false; successful start is not model readiness. |
| embed service stop | No local options. | Runs systemctl stop hades-embedder; command failure propagates, success reports stopped/method. Does not verify shutdown through the configured embedding endpoint. |
| embed gpu status | No local options. | nvidia-smi inventory, device count, cuda_available based on nonempty parsed rows, plus optional embedding device. Provider-info errors become null device. Fixed JSON. |
| embed gpu list | No local options. | Same GPU inventory parser, fixed JSON. Missing nvidia-smi becomes an empty list; a present command exiting nonzero is an error. |
| extract | File, format json (jsonl/table accepted), optional output path. | PDF uses pdftotext; LaTeX uses detex with raw-text fallback on any failure; HTML strips tags/script/style content; text and unknown extensions use UTF-8 read. Reports text/source_path/extension/text_length. No database insertion. |

## Embedding identity and response limits

connect_at overrides only the endpoint; model and expected dimension come from
client defaults (jinaai/jina-embeddings-v4, 2048). The adapter does not apply other
configured model/dimension fields. Embedding responses enforce cardinality,
unique in-range indices, valid vectors and response-model rules before the
adapter indexes the single result. This is stronger than merely accepting HTTP
success. The normal request deadline spans headers and body collection, default
300 seconds. None of these adapters sets an explicit response byte cap.

Provider info looks for the configured model ID to set model_loaded. A sole
nonmatching entry may supply device/dimension metadata while model_loaded stays
false; model_name still reports the configured name. Missing/non-array models
data becomes unloaded metadata rather than a validation error. health_check
accepts any successful info result within five seconds, so available is not proof
that the requested model is loaded. These are source limits requiring private
provider fixtures, not evidence about the active production model.

## Subprocess and file boundaries

systemctl controls the fixed hades-embedder unit even when the configured endpoint
points elsewhere. nvidia-smi CSV rows with fewer than six columns are skipped;
invalid numeric values become null. These subprocess `.output()` calls and PDF/
LaTeX converters have no explicit timeout or output cap in this adapter.

Extraction reads complete text into memory. Its text_length is UTF-8 bytes, not
characters. HTML stripping is a small scanner, not full HTML parsing/entity
decoding. LaTeX raw fallback does not identify converter failure in the result.
PDF/converter arguments use direct process arguments rather than a shell.

If output is supplied, extraction writes/truncates it before validating format.
json writes the data object without an outer envelope; every other requested
format writes plain text, then stdout uses the shared formatter. Thus an invalid
format can fail after changing a file; jsonl output-file contents are plain text,
not the stdout JSONL envelope. No atomic replacement, same-file guard or partial-
write recovery is established here. These are source findings, not executed
file-loss reproductions.

## Remaining evidence

Focused tests remain for adapter/provider readiness semantics, response limits,
converter failures, invalid-format side effects, subprocess lifetime and output
file recovery. Previously retained embedding-client and service lifecycle tests
cover only their stated boundaries; they do not certify every native adapter.
Adding these seven leaves brings source contract tables to 62 of 80; 18 other
leaves, cross-surface parity, quality/provenance and operations remain open.
