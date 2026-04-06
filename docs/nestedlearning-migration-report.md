# NestedLearning Database Migration Report

**Date**: 2026-03-07
**Author**: Claude Opus 4.6 / Todd
**Status**: Ready for cut-over

---

## 1. Problem Statement

The NL knowledge graph — HADES's semantic graph for the NestedLearning research project — had a systemic data quality issue: **all 675 knowledge nodes across 35 collections were missing their canonical `text` field**. Content existed in domain-specific fields (`latex`, `description`, `nl_reframe`, `principles`, etc.), but never in the `text` field that HADES semantic search relies on.

### Impact

- **Double-query pattern**: Agents hitting a node would find embeddings but no readable text, forcing a second semantic search against paper chunks to retrieve the actual content.
- **Token cost inflation**: Only Opus-class models could compensate through multi-step reasoning. Sonnet-class agents couldn't reliably find content at all.
- **Embedding/text incoherence**: The existing node embeddings were computed from `description`/`latex` fields, but there was no canonical `text` representation for agents to read — embeddings pointed to content that wasn't there.

Additionally, the 8 academic papers in NL were ingested from PDFs, introducing rendering artifacts for equations, tables, and algorithms that LaTeX source files would not have.

## 2. What We Fixed

### Phase 1: Backfill `text` fields (PR #131)

Built `hades db backfill-text` CLI command with dry-run support. For each of the 35 knowledge node collections, the command:

1. Identifies nodes with missing or empty `text` fields via AQL
2. Constructs `text` from domain-specific fields using type-aware logic (equations get `name + description + LaTeX`, reframings get `traditional_view + nl_reframe + key_reframing`, etc.)
3. Batch-updates nodes via AQL with whitelist-guarded collection names

**Result**: 675/675 nodes populated. Zero skips — every node had enough content in its domain fields.

### Phase 2: Blue/green database migration

Rather than patching the live NL database in place, we created a clean `NestedLearning` database using a blue/green strategy:

| Step | What | Result |
|------|------|--------|
| **Dump NL** | `arangodump` (109 collections, 58.2 MB) | 0.77s |
| **Restore** | `arangorestore` → NestedLearning | 0.86s |
| **Download LaTeX** | All 8 papers from `arxiv.org/src/{id}` | Saved to `/home/todd/olympus/NestedLearning/papers/` |
| **Flatten** | `latexpand` to resolve `\include`/`\input` | `/tmp/latex-flat/{name}.tex` |
| **Re-ingest papers** | `hades ingest --task code --force` | 127 chunks across 8 papers, with 2048-dim embeddings |
| **Re-embed nodes** | Jina V4 `retrieval.passage` on all 675 nodes | 34s on A6000 (GPU 0) |
| **Verify edges** | 629 equation + 163 smell + 6 definition source edges | All survived dump/restore, all resolve to valid chunks |

### Papers re-ingested from LaTeX source

| Paper | ArXiv ID | Chunks |
|-------|----------|--------|
| Titans | 2501.00663 | 14 |
| HOPE/Connected | 2504.13173 | 14 |
| NL | 2512.24695 | 32 |
| Lattice | 2504.05646 | 20 |
| ATLAS | 2505.23735 | 16 |
| TNT | 2511.07343 | 10 |
| Trellis | 2512.23852 | 9 |
| MIRAS | 2602.24281 | 12 |

## 3. Current Drift: NL vs NestedLearning

The NL database has been actively used since the snapshot was taken. Measured drift:

### Metadata documents: NL has 4 new entries

| Key | Source | Chunks | Timestamp |
|-----|--------|--------|-----------|
| `cold-start-k4-100k-report` | local | 2 | 2026-03-06 21:01 |
| `cold-start-k4-report` | local | 2 | 2026-03-06 22:38 |
| `gpu-tape-summary-spec` | local | 2 | 2026-03-07 17:28 |
| `pyo3-lib-rs` | code | 12 | 2026-03-06 22:38 |

**Total new chunks**: 18 chunks + 18 embeddings in NL that NestedLearning doesn't have.

### No drift in:
- **Collections**: Both have 109 collections (identical set)
- **Knowledge nodes**: 675 nodes — unchanged in both
- **Edge collections**: `nl_smell_compliance_edges` (191), `nl_equation_source_edges` (629), `nl_smell_source_edges` (163), `nl_definition_source_edges` (6) — all identical
- **Hecate specs**: 104 in both
- **Persephone state**: 248 tasks, 1245 logs, 102 sessions — identical
- **Build data**: 12 builds, 1 build_run — identical

### Summary
The drift is **small and well-defined**: 4 code/spec documents with 18 total chunks. No structural or knowledge-graph changes.

## 4. Migration Plan

### Pre-cut-over: Sync the 4 missing documents (5 min)

Re-ingest the 4 drifted documents into NestedLearning. Since they're code/spec files (not papers), this is a simple `hades ingest --task code` operation:

```bash
# These need their source files — check if they exist in the codebase
# then ingest into NestedLearning with the same IDs
poetry run hades --db NestedLearning ingest <source_file> --id <key> --task code
```

For the two `cold-start-*-report` docs (source: `local`), we may need to locate or regenerate the source files. These appear to be training reports that may be ephemeral.

### Cut-over procedure (during build window)

1. **Verify sync**: Confirm NestedLearning has all documents from NL (114 metadata docs)
2. **Update HADES config**: Change the default database for the NL profile from `NL` to `NestedLearning` in `core/config/hades.yaml`
3. **Update agent templates**: Any Hecate or agent prompts that reference `--db NL` should switch to `--db NestedLearning`
4. **Test query**: Run a semantic search on NestedLearning to confirm retrieval works
5. **Freeze NL**: Mark NL as read-only (or just stop using it — no config change needed)
6. **Soak period**: Run Hecate sessions against NestedLearning for 1-2 days
7. **Drop NL**: After soak period confirms no issues, drop the NL database

### Known limitation: Code LoRA vs retrieval embedding mismatch

The paper chunks in NestedLearning were embedded with Jina V4 **Code LoRA** (`task="code"`) because `.tex` files route through the code pipeline. Standard semantic queries use `retrieval.query` embeddings. These are different embedding spaces within the same model.

**Observed impact**: Paper chunk similarity scores are ~0.40 vs ~0.60 for spec/code documents. Content is findable but ranked lower than ideal.

**Future fix options** (not blocking cut-over):
- Add `.tex` to the standard document pipeline's supported extensions (preferred — LaTeX is prose, not code)
- Re-embed paper chunks post-migration with `retrieval.passage` task
- Build a hybrid search that queries both embedding spaces

### Rollback plan

If issues are discovered after cut-over:
1. The NL database remains intact until explicitly dropped
2. Revert config to point back to `NL`
3. No data loss risk — NestedLearning is additive, NL is untouched

---

## 5. What's Better in NestedLearning

| Dimension | NL (old) | NestedLearning (new) |
|-----------|----------|----------------------|
| Knowledge node `text` fields | 0/675 populated | 675/675 populated |
| Node embeddings | From `description`/`latex` (ad hoc) | From canonical `text` (coherent) |
| Paper source | PDFs (rendering artifacts) | LaTeX (native math, tables, algorithms) |
| Paper chunk embeddings | None (raw text, no vectors) | 127 chunks, all with 2048-dim vectors |
| Chunk→node edges | Present but chunk text unembedded | Present, chunks embedded and searchable |
| Agent query cost | Double-query (node → paper search) | Single query (node has text) or graph traversal |
| Sonnet-viable | No (required Opus reasoning) | Yes (text fields remove the gap) |
