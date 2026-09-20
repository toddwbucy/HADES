# CLI training-session release baseline (#115)

At runtime source `64f0686`, the private training alignment fixture now invokes
the real CLI/Rust RPC client against a session-aware CPU provider. Bubblewrap
hides the live `/run/hades` socket and GPU devices, then binds only the private
provider at the expected path. The provider explicitly maps the CLI's required
`cuda:0` request to CPU; this is not GPU validation.

The first full update succeeds and exports three checkpoint-matching vectors.
After one vector is set to null in the disposable database, an immediate
`--new-nodes` invocation fails to acquire a session: `ResourceExhausted`, training
provider has an active session. The fixture does not clear ownership between
commands or shorten the default lease. The private provider and ArangoDB server
are stopped during failure cleanup.

`TrainingSession::drop` spawns release on the current Tokio runtime. The CLI can
return from `block_on` and shut down that runtime before release finishes.
The default 120-second provider lease is fallback recovery; the fixture measures
immediate rejection, not a 120-second outage. No production occurrence is claimed.

[The hash-bound baseline](cli-training-release-baseline.json) retains the actual
failure log and source/probe identities. The opt-in `training_alignment` test is
expected to fail at this baseline. Generate CPU protobuf bindings and use the
private runner's `--training-alignment-python` option as described in the earlier
alignment report. It additionally requires bubblewrap/user namespaces.

Issue #115 requires bounded explicit release at the normal operation boundary,
error/cancellation/clone semantics, and successful back-to-back CLI calls without
fixture workarounds. This report does not claim remediation or deployment.


## Remediation verification

The client now exposes a bounded, awaited `release()`. All clones stop admitting
operations when release begins; concurrent releases share one acknowledgement.
Failed/cancelled attempts remain retryable, and last-drop cleanup/server expiry
remain abnormal-shutdown fallbacks. Callers must finish outstanding operations
before releasing. Both CLI commands await cleanup before returning or printing
success; a cleanup failure does not replace a primary operation error.

[Separate remediation evidence](cli-training-release-remediation.json) retains
the passing private replay and twelve source/probe hashes. Ten core session
contracts pass, including acknowledgement, clones, retry, cancellation, and a
stalled release bounded to five seconds. Focused all-target Clippy passes.

The real CLI fixture performs ten successive lifecycles on one CPU provider with
its unchanged default lease. It verifies full export of three vectors, immediate
missing-only export of one, update failure on a malformed checkpoint, train
failure on a file used as the checkpoint directory, and immediate successful
successors after both failures. Failed operations preserve database rows. An
injected unavailable release acknowledgement after actual provider cleanup
suppresses success output; combined operation/release failure preserves the
operation error and logs cleanup failure. The fixture observes every acquisition
and completed provider release; it never resets ownership between commands.

Full-inference vectors and unselected rows/revisions are compared exactly.
Subset inference compacts the incoming neighbourhood and changes matrix shapes;
its comparison to full inference allows float32 rounding (`rtol=1e-6`,
`atol=1e-7`). The initial exact comparison exposed a maximum difference of
2.24e-8 after the session defect was fixed. This tolerance is not a ranking-quality
threshold. The original failing baseline JSON remains unchanged.

This verifies normal command boundaries and selected failure paths, not a full
successful CLI training run, abrupt process death, GPU behavior, deployment, or
retrieval quality. The optional private fixture is not part of ordinary CI.
