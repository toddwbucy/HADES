# Structural-vector retention after changed text

Runtime source: `8932dac`. The private CLI lifecycle fixture now executes the
parsed-code and parser-free cases previously described by the
[provenance source review](structural-vector-provenance.md).

## Executed cases

The fixture ingests a Python file and a `.legacy` text file using the real CLI,
ArangoDB persistence, parsing and chunking. A private deterministic embedding
peer supplies content vectors; it does not measure model quality. The test then
attaches sentinel `structural_embedding` fields to both file documents and the
parsed symbols, changes both source files, and re-ingests the same root.

| Path | Verified result after changed text |
|---|---|
| Parsed Python | File and replacement symbols no longer contain the old structural vector; an unrelated marker on the replaced file is also removed |
| Parser-free `--unparsed-ext legacy` | File retains its exact prior structural vector and unrelated marker |
| Both | Content hashes change; new chunks contain the changed text and no longer contain the original text |

This confirms the current replacement-versus-merge contract. Parser-free merging
is documented in CLI help; retained fields are not evidence of failed ingestion.
However, a retained structural vector must not be treated as proof that it was
computed from the new content. Coupled with missing-only update behavior, a
successful re-ingestion does not establish structural-vector freshness.

## Reproduction and scope

```sh
python3 scripts/test_isolated_database.py \
  --arangod /path/to/private-arangod --cli-lifecycle
```

The runner owns a disposable server and supplies private database and embedding
endpoints. The new test is included in `codebase_lifecycle` and therefore in the
existing isolated database CI job. See the
[retained result](ingestion-vector-retention-result.json) for execution output,
source hashes and cleanup status.

This is a baseline behavior contract, not a new universal freshness policy.
It does not test document-pipeline replacement, neighboring-node invalidation,
every supported language or adapter, concurrent structural refresh, or retrieval
quality. A uniform generation policy would require deliberate implementation and
updated expectations. No production source, service, corpus or dependency changed.
