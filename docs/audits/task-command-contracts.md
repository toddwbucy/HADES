# Task command contracts

Source cut: `92584154b1272072e81d8cc3946e583aea114a2f`. All 18 native task leaves
were traced through declarations, main, CLI adapters and shared handlers. The
[manifest](task-command-contracts.json) binds seven sources. No task was created,
modified or read from production for this review.

## Shared routing and storage

All adapters parse output format, construct a configured database pool and invoke
shared dispatch. List, show, handoff-show and graph-integration accept --format
(default json; jsonl/table also supported); others use fixed JSON. Results use
the standard envelope. CLI errors propagate to stderr/nonzero. Task storage uses
persephone_tasks, persephone_edges, persephone_logs and persephone_handoffs.
Ordinary missing-task reads are typed errors, but several query-only operations
do not first verify task existence.

List/log limits default to 50/20 and are clamped to 1000 by dispatch; zero is not
rejected. Types accept task/bug/epic; priorities critical/high/medium/low; statuses
open/in_progress/in_review/closed/blocked. These enums do not alone establish
workflow transition enforcement. Native commands use database authority; shared
transport access-tier policy is distinct from business workflow validation.

## Leaf contracts

| Leaf | Inputs/defaults | Result and state semantics |
|---|---|---|
| create | Title; optional description/parent/tags; type task; priority medium. | Trimmed nonempty title, validated type/priority; new open task, task_ plus six random hex digits. Three total conflict attempts. Returns constructed task document; parent existence is not checked here. |
| list | Optional status/type/parent, limit 50, format. | Validates status/type, filters task records, sorts created_at descending; returns tasks/count. Database errors propagate, not missing-collection-as-empty. |
| show | Key, format. | Returns the raw task document, unlike mutation wrappers containing task. Missing task is an error. |
| update | Key; optional title/description/priority; add/remove tags. | Reads current task; validates title/priority, adds unique tags then removes requested tags. Empty patch returns existing; otherwise PATCH then read-back. CLI passes status=None, though shared params support a validated status assignment without transition-table checking. |
| close | Key; optional message. | Rejects already closed, otherwise directly sets closed and optional close_message, then read-back. Does not require in_review or use transition_task. |
| start | Key. | Accepts open or blocked, sets in_progress and clears block_reason, then read-back. Does not use transition_task logging. |
| review | Key; optional message. | Transition in_progress to in_review; optional message is separately best-effort logged. Returns task. |
| approve | Key; human=false. | Transition in_review to closed. The human flag is forwarded but unused; different-reviewer/session guard is explicitly not implemented. |
| block | Key; optional CLI message/blocker. | Handler requires message presence (empty string not rejected). Validates blocker existence and rejects self-blocking before transition; in_progress to blocked, then inserts dependency edge. Duplicate conflict accepted. Edge failure can follow a committed state change. |
| unblock | Key. | Calls transition_task to in_progress and clears block_reason. Transition table permits both open and blocked sources, wider than the blocked-only command comment. Existing dependency edges are retained. |
| handoff | Key; optional CLI message. | Verifies task, requires message presence. Inserts handoff with note; other context arrays empty and session/Git fields null. Three conflict attempts. Inserts handoff_for edge; on edge failure attempts document deletion and warns if cleanup fails. Returns handoff; not a transaction. |
| handoff-show | Key, format. | Latest edge-linked handoff by created_at, or handoff:null. Does not independently verify the task exists. |
| context | Key. | Task, latest handoff, linked sessions, nonclosed blockers, five latest logs. Missing task errors. Sessions are loaded through implements/submitted_review/approved edges. |
| log | Key; limit 20. | Activity records sorted created_at descending, logs/count. No task-existence precheck. |
| sessions | Key. | Edge-linked nonnull sessions sorted started_at descending, with edge_type; task_key/sessions/count. No task-existence precheck or caller limit. |
| dep | Key; optional add/remove; graph=false. | Primary task checked. Add wins over remove and graph; remove wins over graph. Add rejects self/missing target; conflicts treated as idempotent. Remove missing edge returns removed:false. Default lists all blockers, including closed ones; graph adds all blocked_by adjacency edges, not just this task. |
| usage | No options. | Total and aggregates by status/priority/type. Empty query result defaults to zero/empty aggregates; other query failures propagate. |
| graph-integration | Format. | Returns a static four-step compliance protocol. Does not inspect or integrate any graph; adapter still constructs configured pool. |

## Workflow and evidence limits

The common transition helper allows open→in_progress; in_progress→in_review,
blocked or open; blocked→in_progress or open; in_review→closed or in_progress;
and closed→open. It reads, validates, PATCHes, best-effort logs, then reads back.
No revision precondition or encompassing transaction appears on this path.
Close/start and shared task-update have their own rules, so the transition table
is not a universal invariant. Activity logging ignores insert errors and writes
session_key:null. Approval does not prove independent human review.

Dependency mutations do not detect multi-task cycles or enforce dependency
completion when starting/closing. The dep blocked boolean means any returned
edge target, while context excludes closed blockers. Handoff is a note record,
not captured Git/session state. These source mismatches and partial-write,
concurrent-update, conflict-classification and malformed-response boundaries need
focused fixtures before operational findings or guarantees are claimed.

This map adds 18 leaves to the 33 database and four system/analyzer leaves:
55 of 80 have source contract tables, with 25 other leaves and cross-surface
parity still open. Source coverage is not 55 end-to-end tests, deployed workflow
certification or completion of epic #12.
