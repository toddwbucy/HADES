# WeaverTools writer failure propagation (#68)

The original writer returned exit 0 and printed an attempted-row success count
after an import reported rejected documents, provided its report write succeeded.
Collection errors were ignored; failed cursor pages could become empty/partial
scope. An in-memory one-row reproduction verified that behavior without a server.

The writer now validates complete scope pages before extraction or writes,
rejects ambiguous path/key pairs, checks collection names/types, and tolerates
only a duplicate-name error followed by compatible collection properties.
A failed cursor continuation attempts deletion of its validated cursor ID;
server TTL remains the fallback if cleanup itself fails.

Imports require nonnegative integer counters, zero errors/ignored/empty rows,
and created-plus-updated counts equal to the submitted rows. HTTP failure status
cannot be overridden by a JSON body claiming success. Report writes need the
expected document identity. Only acknowledged imports contribute to the final
count. Failures exit nonzero and state that earlier writes may persist.

This is not a whole-run transaction, a stale-record retirement algorithm or a
change to identifier derivation. A later collection/import/report failure can
leave earlier writes committed. An earlier `latest` report may remain after a
failed run; callers must honor the failure exit status and must not treat it as
certification of the attempted run.

The focused suite has 52 passing cases including scope, existing/wrong-type
collections, HTTP errors, partial/malformed imports, report errors, normal success
and redirect policy. These use synthetic responses and private HTTP peers.
Real ArangoDB acceptance of the strengthened response contracts still requires
an isolated fixture; no production graph was used.
