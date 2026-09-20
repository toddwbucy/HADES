# Database access boundary (#74)

## Active deployment evidence

On 2026-09-20, `arangod.service` was active at PID 3014 with zero restarts since
September 17. Its selected argv uses the per-user configuration and has no
explicit authentication override. That configuration predates process startup,
sets `server.authentication = false`, and exposes `127.0.0.1:8529` plus the
per-user Unix socket. One credential-free `GET /_api/database` returned HTTP 200,
`error=false` and three database entries. Only status and count were published.
The request had a three-second socket timeout and a 64 KiB response cap.

This confirms local administrative metadata access without credentials. It does
not establish remote reachability or exercise production write permissions.
Unix socket modes alone cannot protect a separate TCP listener. Database-user
grants are not an enforced isolation boundary when authentication is disabled.
Issue #74 is a P2 deployment finding requiring a documented local trust policy
or authenticated access; no unauthorized use or data loss was observed.

## Repository contract

`DatabaseConfig` has one username/password pair. `ArangoClient::from_config`
uses that same pair for both clients, while its `read_only` argument selects the
socket. Thus `reader()` is a routing convention, not a read-only database identity.
The configuration comment implying separate read/write grants has been corrected.
The current bootstrap script deliberately grants the named DBA user broad access
on `_system` and future databases; it is not a least-privilege application-user
provisioner. Existing-user grants are preserved when it is rerun.

The HADES transport's command tiers and database allowlist remain necessary even
with database authentication enabled: requests executed with one service identity
share that identity's database permissions. Distinct direct database consumers can
have distinct users/grants, as the following fixture demonstrates. It does not
add separate reader/writer credentials to HADES or certify its complete transports.

## Private authenticated matrix

The [recorded result](database-acl-result.json) comes from a fresh ArangoDB 3.12.11
server with authentication explicitly enabled for its only listener, a private
Unix socket. Generated JWT bootstrap material and passwords remained private.
Two synthetic users, two synthetic databases and ordinary/protected collections
exercise these cases:

| Request | Expected and observed |
|---|---|
| Missing or wrong credentials | 401 in both cases |
| Reader fetch / insert | 200 / 403 |
| Writer fetch / insert | 200 / 202 |
| Writer insert into collection with explicit read-only override | 403 |
| Each user's access to the other database | 401 in both cases |
| Each user's administrative database listing | 401 in both cases |

All 11 checks passed. Administrative follow-up reads confirm both rejected inserts
left no document and the allowed insert is present. The first probe expected 403
for database-level denial; the observed contract was 401. The retained run uses
that exact status, with 403 still required for denied writes in an accessible
database. Both runs cleaned up their owned server. No live endpoint was used.

Repeat using existing matching binaries and adjacent ICU/timezone data:

```bash
PYTHONDONTWRITEBYTECODE=1 timeout 150 python3 scripts/verify_database_acl.py \
  --bin-dir /path/to/existing/arangodb/bin
```

The opt-in probe installs nothing and accepts no server URL. It uses a new 0700
artifact directory, one CPU, low priority, small server caches, an 8 GiB server
address-space limit, one owned process group and explicit cleanup. The recorded
run used the private staged binaries from the historical restore rehearsal.
This matrix is not a production credential inventory, a TCP test, a full
administrative-operation matrix or an application deployment test. It is not part
of the default CPU unit-test suite.

## Reviewable access-change plan

1. Identify every actual database consumer and the databases/collections it needs.
   Distinguish the HADES DBA/provisioning role from ordinary application readers
   and writers. Do not apply the fixture's example grants blindly to production.
2. Prepare dedicated credentials and explicit grants in protected configuration,
   including future-database defaults. Verify all required workflows privately,
   including installer/provisioner behavior and transport allowlists. Record any
   broad administration grants and their owner justification.
3. Arrange the maintenance window with the owner. Before any operation risking
   data loss, take and verify a snapshot covering the effective active data and
   required configuration. Specify the database-consistency method and a rollback
   procedure; a legacy-dataset snapshot does not cover `dbpool/home/todd`.
4. In that separately authorized window, apply the reviewed authentication/client
   configuration together. Verify missing/wrong credentials fail on both actual
   listener types, permitted application workflows work, and unintended database
   scopes are denied. Exercise destructive cases only on disposable fixtures.
5. Retain the pre-change snapshot through verification, record the installed
   artifact/configuration identifiers and check monitoring. If validation fails,
   stop the rollout and execute the approved rollback plan.

Alternatively, the owner may explicitly accept a trusted-local-process boundary
with a rationale and documented TCP/Unix exposure. No such acceptance, credential
provisioning, snapshot, configuration change or restart has been performed by this
review. Routine backup coverage and incident recovery objectives remain separate
open work in #63.
