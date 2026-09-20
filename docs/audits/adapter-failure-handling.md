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
count. Failures exit nonzero and state that earlier writes may persist. Valid partial
import counters and earlier successful batches survive later failures as a
reported lower bound on acknowledged imported rows. Lost responses may hide
additional writes; the count is not an exact persistence inventory. Dangling
endpoint failures also print this warning and never print the success line.

This is not a whole-run transaction, a stale-record retirement algorithm or a
change to identifier derivation. A later collection/import/report failure can
leave earlier writes committed. An earlier `latest` report may remain after a
failed run; callers must honor the failure exit status and must not treat it as
certification of the attempted run.

The focused suite has 59 passing cases including scope, existing/wrong-type
collections, HTTP errors, partial/malformed imports, report errors, normal success
and redirect policy. These use synthetic responses and private HTTP peers.
The [recorded real-database result](adapter-database-result.json) also passed on
ArangoDB 3.12.11: initial and repeated writes return success, schema-rejected
imports return failure, and existing collections of the wrong type are rejected.
The repeat run exercises real duplicate-name responses and collection properties;
imports exercise actual ArangoDB acknowledgement fields. No production graph was
used.

Reproduce with an existing binary:

```bash
PYTHONDONTWRITEBYTECODE=1 timeout 90 python3 scripts/verify_adapter_database.py \
  --arangod /path/to/existing/arangod
```

This opt-in probe creates a fresh server on a private Unix socket and an owned
loopback HTTP bridge for the adapter. It uses synthetic records, one CPU, nice 10,
an 8 GiB server address-space cap and small database caches. Configuration and
credentials are private, server groups are cleaned up, and artifacts remain in a
new temporary directory. Extraction is stubbed to isolate the writer contract;
this does not certify a complete WeaverTools extraction/ingestion run. The probe
is retained separately from the normal CPU test suite.
