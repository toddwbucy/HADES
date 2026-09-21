# Python import target correctness

Tracks epic #12 and issue #149. Production databases and services were not accessed.

## Reproduction

At source `6984f125ae6158dec30bdfa6e106ec84e1c3d173`, the actual scoped Python import resolver accepted a bare-name match outside the requested module. The isolated fixture supplies synthetic import and definition symbols in `/fixture/repository`, then calls the production index builder and resolver directly.

| Input | Baseline | Required behavior |
|---|---|---|
| External `from external_package import Config`; only unrelated local definition | Symbol edge to `unrelated.py`, resolved=true | No local edge |
| `from config import Config`; config.py lacks the definition | Symbol edge to `unrelated.py`, resolved=true | Existing unresolved file edge to config.py |
| config.py defines Config, alongside unrelated definition | Correct symbol edge to config.py | Preserve correct symbol edge |

The regression test failed with exit 101; both erroneous cases were observed before the final assertion. The valid control passed. `python-import-targets-baseline.json` preserves observations and the fixture source hash at baseline commit `61ecc57`; no live-corpus corruption is asserted.

## Correction and verification

Symbol resolution now requires the candidate to belong to the resolved target module. It cannot fall back to the first unrelated bare-name match. Known-module file fallback and valid symbol resolution remain supported.

Run with the repository toolchain:

```bash
cargo test --offline -p hades-cli --bin hades python_import -- --nocapture
cargo clippy --offline -p hades-cli --all-targets -- -D warnings
```

The focused suite covers seven tests, including package `__init__` imports, deduplication, file fallback and the three new cases. See the remediation manifest for verified outcomes and file hashes.

## Limits and remaining work

This is a resolver-level regression, not a CLI/parser/database integration reproduction. Relative imports, re-exports, module-prefix fallback, duplicate module paths and ambiguous definitions need separate conformance coverage. No quality score follows from these tests. Previously stored edges are not repaired automatically: production re-ingestion or cleanup needs a separately reviewed plan, effective-data snapshot and appropriate service coordination. The full epic remains open.
