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
