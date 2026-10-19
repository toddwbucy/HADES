# Long-document embedding windows

HADES plans context windows from the running backend's `max_seq_length`, rather
than a workstation or GPU name. Parsed code ingestion uses late chunking by default; parser-free code keeps
its existing independent embedding path.
Document ingestion keeps independent chunk embedding by default; opt in with:

```yaml
chunking:
  document_late_chunking: true
  prechunk:
    overlap_tokens: 1000
    safety_margin_tokens: 256
    max_window_tokens: null
```

These are installation-wide settings today. The
[per-database profile audit and plan](specs/database-embedding-profiles.md)
addresses #215 and proposes the implementation sequence for #225.
The opt-in requires a backend implementing HADES's `late_chunk.boundaries`
extension. Unsupported or malformed responses fail explicitly.

## Window placement and passage identity

The usable token budget is the backend ceiling minus the safety margin, limited
by `max_window_tokens` when configured. The initial byte budget uses the existing
two-bytes-per-token estimate. This is **not** a tokenizer measurement or a promise
that the window fills the model's token context. The service checks actual input
size, including its task prompt and special tokens, before inference.

For a covered source span of length `L`, byte budget `W`, and minimum byte overlap
`O`, an ASCII plan uses one window when `L <= W`; otherwise it uses
`1 + ceil((L - W) / (W - O))`. Starts are spread evenly from the source's first
passage to its last. The final window is anchored at the end, avoiding a short
tail. UTF-8 boundaries can shorten windows by up to three bytes and require
additional windows. An impossible budget/overlap fails instead of dropping text.

Each retrieval passage selects the containing window whose center is closest
to the passage center; ties select the earlier window. Only that window pools
the passage, so overlapping context does not create duplicate embedding rows.
Windows with no selected passages need no inference request. Existing overlap
between retrieval chunks is a separate setting and remains intact.

A passage that no window contains is divided into adjacent fragments until each
fits. Splitting adds no passage overlap. Each fragment retains its original
`parent_chunk_index`; stored `chunk_index` remains contiguous in file order.
Unsplittable UTF-8 input fails. Byte offsets remain source offsets; the HTTP
client converts them to character offsets only when constructing pooling ranges.

Inputs already fitting one late-chunking window retain their existing boundaries
and context span. A verified `PE_INPUT_TOO_LARGE` response refines the byte budget
and replans the whole span. All provisional vectors are discarded before retry;
no partial plan reaches storage. There are at most 32 planning attempts and fewer
than 100,000 windows. A temporary transport/server failure gets one exact retry;
invalid vectors, model drift and unsupported late chunking do not trigger splitting.

## Provenance and explicit re-ingestion

Document metadata and code file records contain `embedding_window_plan`: the
observed backend ceiling, usable token budget, initial/effective byte budgets,
minimum byte overlap, refinement count, source window ranges and each stored
chunk's selected window. Embedding rows contain `embedding_window`, with the
selected range and a compact plan summary including window count and actual
minimum/maximum byte strides and overlaps. Model identity remains on each vector.

For example, a 40,000-unit span at a 32,512-unit budget needs two windows. At an
11,644-unit budget and 1,000-unit minimum overlap it needs four. Raising that
minimum to 6,000 needs **seven**, not six: six would provide only about 5,973 units
of overlap. These are planner examples, not measured tokenizer or quality results.

Changing a ceiling or enabling document late chunking does not automatically
rewrite unchanged sources. Review the source and explicitly re-ingest it:

```bash
hades --database YOUR_DATABASE ingest /path/to/document.pdf --force
```

Existing records without provenance remain unknown; do not infer their original
window plan. No automatic re-embedding queue or cross-database migration is added.
Context windows do not connect a long document's distant beginning and end; graph
relationships still carry those connections.

## Validation boundaries

Service-free tests cover fitting/just-over-budget inputs, evenly distributed
windows, nearest-center ties, multiple ceilings, minimum overlap and Unicode
source coverage. HTTP fixtures exercise size refusals, malformed vectors, model
drift, exact parent/index mapping and failure before replacement. Disposable
database contracts verify document and code provenance and retained committed
records after failed preparation.

These checks establish mapping and persistence correctness. They do not measure
retrieval quality, select an optimal overlap, or prove token-exact window filling.
Compare representative papers and retrieval judgments before choosing a larger
overlap for routine ingestion. No active research documents are re-embedded by
this change or its tests.
