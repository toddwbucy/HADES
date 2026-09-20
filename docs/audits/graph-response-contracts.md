# Graph response validation audit

At source `a563d843b63eaad15c9c4d563fa366dcfba0b47b`, an actual CLI against a
private Unix HTTP peer accepted `{}` as the successful Gharial graph-list reply.
It exited zero with `success: true` and `data.graphs: []`. Valid empty and
populated replies passed controls first. The regression assertion then failed as
expected; this is not a production incident or a claim about normal ArangoDB output.

The shared `db_graph_list` handler defaults a missing/non-array `graphs` field
to an empty array, and missing graph names to `unknown`. Callers cannot distinguish
malformed metadata from an empty database. Automatic graph resolution also uses
this handler and can consequently report that no graph exists.

[Baseline evidence](graph-response-baseline.json) records source/probe hashes and
observed output. Reproduce in an isolated checkout with:

```sh
cargo test -p hades-cli --test db_response_shapes graph_list_rejects_malformed_success_responses -- --nocapture
```

Required remediation: validate graph-list structure at the shared boundary,
propagate malformed metadata as an error, retain valid empty/populated results,
and verify CLI failure plus daemon error envelopes and graph-resolution behavior.
Do not infer graph absence from a malformed reply. Graph creation/drop responses,
format flags and materialization partial failures remain separate audit scope.
