# Deployment provenance follow-up (2026-09-20)

Bounded read-only inspection found `hades-daemon.service` active with PID 4240,
one recorded restart, and start timestamp 2026-09-17 20:15:31 CDT. Its running
`/proc/4240/exe` and installed `/home/todd/.local/bin/hades` both hash to
`993712814319be5f663a3c98fdb51b13c27a144def1adcb81c1178109af986ac`.
These agree with the initial audit baseline; no additional daemon restart is
shown by these counters.

The installed ELF contains GNU build ID
`319ffd69c4d1ce2ba7e56988fb5e68775529032f` and a compiler comment identifying
rustc 1.98.1 (`48a229cea`, 2026-09-01). An ELF build ID identifies an artifact;
it is not evidence of the HADES Git commit used to compile it. The compiler's
commit and embedded LLVM revision likewise identify toolchain components only.

The source search for VERGEN/GIT_SHA/BUILD_COMMIT/COMMIT_HASH and Git-revision
commands in `crates/`, `scripts/`, `deploy/` and `Cargo.toml` found no revision
embedding hook. A filename inventory under the installed binary directory,
`~/.config/hades`, repository deployment and script directories found no file
named for a release/build manifest or provenance record. This bounded search
does not prove no external build record exists.

Exact running HADES source revision remains unverified. A verifiable mapping
requires a retained build manifest or independently reproducible artifact with
matching hash, source revision/dirty state, toolchain and dependency provenance.
The current checkout's Git revision and package version cannot substitute for
that mapping. Repository remediation merges are not deployment evidence.

No binary was executed or replaced, no service was restarted, and no production
configuration, credentials, database contents or model data were read or changed.
