# Viewer backend envelope validation (#107)

Baseline source: b48060ae08703d9ff90be6cadccad26a0ce581c6. The
[retained reproduction](repros/viewer_failure_envelope.rs) is a historical module
to append to server.rs only in an isolated baseline checkout. Its
[result](viewer-envelope-baseline.json) pins the source and probe hashes.

## Confirmed behavior and impact

An actual private backend child exits zero and prints a graph envelope. The
in-process viewer /api/graphs route receives both a success:true control and an
explicit success:false envelope with graph-shaped data. Both baseline responses
are HTTP 200 with the synthetic graph name; one fixture passed in 0.04 seconds.

The backend-process boundary checks exit status, while assemble.rs parsed data
without checking the envelope discriminator. A failed/contradictory backend
result could therefore be displayed as usable graph data. This is a P2 response
contract defect tracked in #107. It does not prove the current production CLI
emits this contradictory result or that a user saw an incorrect production graph.

## Fix and compatibility boundary

All seven JSON envelope consumers now share parse_success_envelope: database
listing, graph listing, collection counts, neighbors, a single node, AQL documents
and AQL chunk text. The helper requires an object, accepts success:true, rejects
success:false, and rejects non-Boolean success values before reading data.

An absent success field remains supported for legacy backend compatibility.
This is an explicit compatibility policy, not strict enforcement of a new
protocol version. Successful missing-node results retain their existing behavior.
Raw JSONL exports do not use this helper and preserve their existing parser.

Errors use a command label and generic failure description. Raw backend error
text and graph data from failed envelopes are not copied to browser errors.
The existing route maps parser failures to HTTP 502.

## Validation

The [remediation manifest](viewer-envelope-remediation-result.json) binds the
changed source. All 29 hades-viewer unit/router tests pass (1.72 seconds).
All-target Clippy with warnings denied passes after naming the test parser
function-pointer type; that lint correction does not change behavior. Coverage includes:

- The actual private-child route: success control returns 200; explicit failure
  returns 502 without the private diagnostic or synthetic graph name.
- Every one of the seven parsers: true accepted; false, string, null and numeric
  success rejected; legacy omission accepted; non-object and invalid JSON rejected.
- Existing missing-node, JSONL, graph assembly, admission/deadline/process ownership
  and response-budget behavior.

The historical baseline and production source remain untouched. Tests use private
temporary processes and in-process routes, with no live backend or database.
Final CI and merge remain required; no deployment or browser asset certification
is implied by these tests.
