#!/usr/bin/env bash
#
# Install the managed (model-bound) workspace on the operator's laptop (#225).
#
# Carries out Part C of docs/specs/db-setup-door.md as ten ordered steps: install the
# main build, register the Jina snapshot, configure the provider and the setup ledger,
# switch the embedder to the verified provider unit, create the managed database with
# `hades db setup`, back it up, move the reading desk into it, ingest the first corpus,
# and back it up again. docs/installation-thinkpad.md, "Managed workspace", says what to
# look at after each step.
#
# Every step tests its own postcondition first and reports "already done" without
# acting, and refuses in one line when a precondition is missing. Nothing is forced,
# deleted or overwritten silently: a file that exists with other content is a refusal,
# not a replacement. Re-running `all` after a failure resumes at the first undone step.
# The script never drops the source database; after the last step it prints the
# retirement gate and stops.
#
# Usage:
#   scripts/install/thinkpad-managed-workspace.sh [--dry-run] [--yes] STEP
#
# Steps, in order (`all` runs every undone one and stops at the first failure):
#   check            read-only: values, preconditions and each step's state
#   install-main     install the main build's executables; bring the checkout to it
#   register         register the approved Jina snapshot in the model registry
#   configure        registry and ledger roots, the YAML sections, the unit's env file
#   switch-embedder  install and start the verified provider unit; disable the legacy one
#   setup-db         `hades db setup run`, then inspect and diagnose
#   backup-1         first combined backup (database, setup ledger, registry)
#   move-desk        final source backup, then move the reading desk into the database
#   ingest           ingest the corpus's Markdown and text files; ask two questions
#   backup-2         second combined backup, verified in isolation
#
# --dry-run prints every command a step would run, with resolved values, and runs
# nothing. --yes skips the confirmation that install-main, switch-embedder and move-desk
# ask before they stop or switch services.
#
# Values come from the block below; each can be overridden by its HADES_MW_* variable.
# Each run that acts records its evidence in a new private directory,
# $DATA/managed-workspace/<UTC timestamp>/, where DATA is the resolved data directory;
# step completion markers live in $DATA/managed-workspace/state/. The record holds
# copies of configuration with credentials, so it stays private.
#
# Status:
#   Written before its first run (2026-11-10). The steps that need no service were
#   run against scratch paths; the service steps are exercised only by the operator's
#   run. The private backup driver must accept --database before backup-1.

set -euo pipefail

# Values ------------------------------------------------------------------------------

DB=${HADES_MW_DATABASE:-Conveyance_KG_v2}
SOURCE_DB=${HADES_MW_SOURCE_DATABASE:-Conveyance_KG}
PROVIDER=${HADES_MW_PROVIDER:-laptop-jina}
INSTALLATION_ID=${HADES_MW_INSTALLATION_ID:-laptop-2026-11}
OPERATION_ID=${HADES_MW_OPERATION_ID:-conveyance-kg-v2-1}
# A label; a retry repeats it exactly, because the ledger admits only an identical request.
GENERATION=${HADES_MW_GENERATION:-2026-11-10-g1}
CONTEXT_CAP_TOKENS=${HADES_MW_CONTEXT_CAP_TOKENS:-16384}
WINDOW_OVERLAP_TOKENS=${HADES_MW_WINDOW_OVERLAP_TOKENS:-1000}
PASSAGE_TOKENS=${HADES_MW_PASSAGE_TOKENS:-500}
OVERLAP_TOKENS=${HADES_MW_OVERLAP_TOKENS:-200}
MAX_INPUT_TOKENS=${HADES_MW_MAX_INPUT_TOKENS:-20000}
MODEL_ID=jinaai/jina-embeddings-v4
REVISION=${HADES_MW_REVISION:-853c867b65b749f3c3c72a06868140d842e04f06}
MANIFEST_MD5=${HADES_MW_MANIFEST_MD5:-446160668af00248832ed8cf5d8d4432}
MAIN_COMMIT=${HADES_MW_MAIN_COMMIT:-09df08febe62570a883563030e9c1cf4d9f9aabc}
HADES_SHA256=${HADES_MW_HADES_SHA256:-633b68a1a9c1ce919d2c22ac227e07cb48ccb5bc071ff1926057e9ab44774b4c}
VIEWER_SHA256=${HADES_MW_VIEWER_SHA256:-22ac0c56a12a16b79af8f7c2c84ae9116efbeaa6a5f5e825559f30f3f494131f}
# The f930fbc executables installed on 2026-10-19, which the rollback restores.
PRIOR_HADES_SHA256=${HADES_MW_PRIOR_HADES_SHA256:-cf0573ebbb42893cb7f1ccf5f2a69caf506df57cbe422e0fcd705e31c3ec6b42}
PRIOR_VIEWER_SHA256=${HADES_MW_PRIOR_VIEWER_SHA256:-6efac5881ed6b5ea7ac30c0a7a6131979e6b224fc14618104d9a354c6cac2c72}

# Paths, from $HOME and the resolved data directory. The provider catalog, the registry
# and the setup ledger refuse a path with a symlink ancestor, and ~/.local/share/hades
# is a symlink, so everything they read is written from the resolved DATA.
CHECKOUT=${HADES_MW_CHECKOUT:-$HOME/Projects/HADES}
DATA_LINK=${HADES_MW_DATA:-$HOME/.local/share/hades}
DATA=$(readlink -f "$DATA_LINK")
CONFIG_DIR=${HADES_MW_CONFIG_DIR:-$HOME/.config/hades}
UNIT_DIR=${HADES_MW_UNIT_DIR:-$HOME/.config/systemd/user}
BIN_DIR=${HADES_MW_BIN_DIR:-$HOME/.local/bin}
BUILD=${HADES_MW_BUILD:-$HOME/Projects/hades-targets/deploy-main/release}
ARANGO_HOME=${HADES_MW_ARANGO_HOME:-$HOME/.local/opt/arangodb-3.12.12.1}
SNAPSHOT=${HADES_MW_SNAPSHOT:-$DATA/models/jina-v4/$REVISION}
CORPUS=${HADES_MW_CORPUS:-$CHECKOUT/docs}
RECORDS=${HADES_MW_RECORDS:-$DATA/managed-workspace}

REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
HELPER=$REPO/scripts/install/managed_workspace.py
TEMPLATE=$REPO/deploy/systemd/hades-embedder-verified.service
HADES=$BIN_DIR/hades
DAEMON_ENV=$CONFIG_DIR/daemon.env
VERIFIED_ENV=$CONFIG_DIR/embedder-verified.env
ROOT_PASSWORD_FILE=$CONFIG_DIR/arango-root-password
REGISTRY=$DATA/model-registry
LEDGER=$DATA/setup-ledger
RUN_DIR=$DATA/run/verified
SOCK=$RUN_DIR/provider.sock
ARANGO_SOCK=$DATA/run/arango.sock
DAEMON_SOCK=$DATA_LINK/run/hades.sock
UNIT=$UNIT_DIR/hades-embedder-verified.service
DAEMON_DROPIN=$UNIT_DIR/hades-daemon.service.d/managed-database.conf
VIEWER_DROPIN=$UNIT_DIR/hades-viewer.service.d/managed-database.conf
LOCAL_INSTALL=$CHECKOUT/.local-install
DRIVER=$LOCAL_INSTALL/backup-workspace.py
LATEST=$LOCAL_INSTALL/latest-workspace-backup.txt
STATE=$RECORDS/state
RECORD=$RECORDS/$(date -u +%Y%m%dT%H%M%SZ)

# The two first-ingest questions, each with the corpus-relative sources that answer it.
QUERY_INVENTORY="which collections does the setup coordinator's inventory accept"
EXPECT_INVENTORY=(specs/recoverable-workspace-setup.md specs/db-setup-door.md)
QUERY_CEILING="what sequence ceiling does the laptop embedder run with and how was it chosen"
EXPECT_CEILING=(installation-thinkpad.md installation-thinkpad-services.md
  specs/db-setup-door.md)

STEPS=(install-main register configure switch-embedder setup-db backup-1 move-desk
  ingest backup-2)
DRY_RUN=0
YES=0
CURRENT=""

# Output and command plumbing ---------------------------------------------------------

# Show paths as $DATA and ~, never as the machine's own directory names.
display() {
  local text=$1
  # shellcheck disable=SC2016  # the literal word $DATA is the point
  text=${text//"$DATA"/'$DATA'}
  text=${text//"$HOME"/'~'}
  printf '%s' "$text"
}

say() { printf '%s\n' "$(display "$*")"; }

join() {
  local IFS=';'
  local text="$*"
  printf '%s' "${text//;/; }"
}

# The resolved data directory itself, in ~ form when it is under the home directory.
data_location() {
  if [[ $DATA == "$HOME"/* ]]; then printf '~%s' "${DATA#"$HOME"}"; else printf 'outside ~'; fi
}

# Quote one word for display: plain words and <placeholders> as they are, anything else
# in single quotes.
quote() {
  local word=$1 safe='^[][A-Za-z0-9_./:@%+=,~^$-]+$'
  if [[ $word =~ $safe || $word == *"<"*">"* ]]; then
    printf '%s' "$word"
  else
    printf "'%s'" "${word//\'/\'\\\'\'}"
  fi
}

# Print a command before it runs; SUFFIX, when set, is a redirection shown as written.
show() {
  local line="+" word
  for word in "$@"; do line+=" $(quote "$word")"; done
  [[ -z ${SUFFIX:-} ]] || line+=" $SUFFIX"
  say "$line"
  if ((!DRY_RUN)) && [[ -d $RECORD ]]; then
    printf '%s\n' "$(display "$line")" >>"$RECORD/commands.log"
  fi
}

# Refuse in one line. Under --dry-run, report it and carry on printing commands.
refuse() {
  if ((DRY_RUN)); then
    say "dry-run: $CURRENT would refuse now: $*"
    return 0
  fi
  printf '%s\n' "$(display "$CURRENT: $*")" >&2
  exit 1
}

# Pass a command's terminal output through display(), so it too shows $DATA and ~.
filter() {
  local line
  while IFS= read -r line || [[ -n $line ]]; do say "$line"; done
}

run() {
  show "$@"
  if ((!DRY_RUN)); then "$@" 2>&1 | filter; fi
}

# Run with stdout into a file; the command's exit status must be zero.
run_to() {
  local file=$1
  shift
  SUFFIX="> $file" show "$@"
  if ((!DRY_RUN)); then "$@" >"$file" 2> >(filter >&2); fi
}

# Run with stdout into a file and keep the exit status in STATUS instead of failing.
run_to_status() {
  local file=$1
  shift
  SUFFIX="> $file" show "$@"
  STATUS=0
  if ((!DRY_RUN)); then "$@" >"$file" 2> >(filter >&2) || STATUS=$?; fi
}

# Run with stdout shown and kept in a file.
run_tee() {
  local file=$1
  shift
  SUFFIX="| tee $file" show "$@"
  if ((!DRY_RUN)); then "$@" 2>&1 | tee "$file" | filter; fi
}

helper() { python3 "$HELPER" "$@"; }

# Write a new file from a generator function, privately and atomically. Under --dry-run,
# print the content instead.
write_file() {
  local mode=$1 file=$2 generator=$3
  say "+ write $file (mode $mode) from $generator:"
  "$generator" | sed 's/^/    /' | filter
  ((DRY_RUN)) && return 0
  [[ ! -e $file ]] || refuse "$file exists; not overwriting it"
  "$generator" >"$file.mw-new"
  chmod "$mode" "$file.mw-new"
  mv -n "$file.mw-new" "$file"
  [[ ! -e $file.mw-new ]] || refuse "$file appeared during the write; left $file.mw-new"
}

append_file() {
  local file=$1 generator=$2
  say "+ append to $file from $generator:"
  "$generator" | sed 's/^/    /' | filter
  ((DRY_RUN)) && return 0
  "$generator" >>"$file"
}

confirm() {
  ((DRY_RUN || YES)) && return 0
  [[ -t 0 ]] || refuse "needs confirmation; run it from a terminal or pass --yes"
  local answer
  read -r -p "$1 [y/N] " answer
  [[ $answer == y || $answer == yes ]] || refuse "not confirmed; nothing was changed"
}

# The record directory exists only once a step acts, so check and --dry-run write nothing.
begin_record() {
  ((DRY_RUN)) && return 0
  install -d -m 700 "$RECORDS" "$STATE" "$RECORD"
}

mark() {
  local name=$1 value=$2
  say "+ mark $name done: $value"
  if ((!DRY_RUN)); then printf '%s\n' "$value" >"$STATE/$name"; fi
}

load_env() {
  if [[ -f $DAEMON_ENV ]]; then
    set -a
    # shellcheck disable=SC1090  # the operator's file, not part of this repository
    . "$DAEMON_ENV"
    set +a
  fi
  # daemon.env names the YAML; an explicit override wins, for scratch runs.
  YAML=${HADES_MW_YAML:-${HADES_CONFIG:-$CONFIG_DIR/hades.yaml}}
  export HADES_CONFIG=$YAML
}

need_env() {
  [[ -f $DAEMON_ENV ]] || refuse "daemon.env is missing; it supplies the database account"
}

sha_is() { [[ -f $1 ]] && [[ $(sha256sum "$1" | cut -d' ' -f1) == "$2" ]]; }

# Check executables against expected SHA-256 values: pairs of hash and path.
check_hashes() {
  while (($#)); do
    say "+ sha256sum -c: $1  $2"
    if ((!DRY_RUN)); then
      printf '%s  %s\n' "$1" "$2" | sha256sum -c --quiet - ||
        refuse "$(basename "$2") does not match its expected SHA-256"
    fi
    shift 2
  done
}

private_dir() {
  [[ -d $1 && ! -L $1 ]] && [[ $(stat -c '%a %u' "$1") == "700 $UID" ]]
}

avail_gib() { df --output=avail -B1G "$1" | tail -1 | tr -d ' '; }

unit_enabled() { systemctl --user is-enabled --quiet "$1" 2>/dev/null; }
unit_active() { systemctl --user is-active --quiet "$1" 2>/dev/null; }

# Generators ----------------------------------------------------------------------------

render_yaml_block() {
  cat <<EOF

registered_providers:
  registry_root: $REGISTRY
  providers:
    - id: $PROVIDER
      socket: $SOCK
setup:
  ledger_root: $LEDGER
  installation_id: $INSTALLATION_ID
EOF
}

# An idle unload would invalidate every receipt a client holds, so the timeout is 0.
render_verified_env() {
  printf 'HADES_EMBEDDER_IDLE_TIMEOUT=0\nHADES_EMBEDDER_MAX_TOKENS=%s\n' "$MAX_INPUT_TOKENS"
}

render_unit() {
  sed -e "s|@SERVICES_DIR@|$CHECKOUT/services|g" \
    -e "s|@REGISTRY_ROOT@|$REGISTRY|g" \
    -e "s|@PROVIDER_ID@|$PROVIDER|g" \
    -e "s|@SOCKET_PATH@|$SOCK|g" "$TEMPLATE"
}

# Drop-ins change only the database arguments; the daemon keeps its --mcp-db-prefix.
render_daemon_dropin() {
  printf '[Service]\nExecStart=\n'
  grep '^ExecStart=' "$UNIT_DIR/hades-daemon.service" |
    sed -e "s/--db $SOURCE_DB /--db $DB /" -e "s/--mcp-dbs $SOURCE_DB /--mcp-dbs $DB /"
}

render_viewer_dropin() {
  printf '[Service]\nExecStartPre=\nExecStart=\n'
  grep -E '^ExecStart(Pre)?=' "$UNIT_DIR/hades-viewer.service" |
    sed -e "s/\b$SOURCE_DB\b/$DB/g"
}

# Step states: "done", "pending: why" or "blocked: why". Read-only. ----------------------

installed_build() {
  sha_is "$BIN_DIR/hades" "$HADES_SHA256" && sha_is "$BIN_DIR/hades-viewer" "$VIEWER_SHA256"
}

installed_prior() {
  sha_is "$BIN_DIR/hades" "$PRIOR_HADES_SHA256" &&
    sha_is "$BIN_DIR/hades-viewer" "$PRIOR_VIEWER_SHA256"
}

# The verified unit runs python from the checkout's services tree, so that tree must
# equal the main commit, committed and in the working tree.
checkout_at_main() {
  git -C "$CHECKOUT" merge-base --is-ancestor "$MAIN_COMMIT" HEAD 2>/dev/null &&
    git -C "$CHECKOUT" diff --quiet "$MAIN_COMMIT" HEAD -- services &&
    [[ -z $(git -C "$CHECKOUT" status --short -- services) ]]
}

state_install_main() {
  local why=()
  if ! installed_build; then
    installed_prior || {
      echo "blocked: installed executables are neither f930fbc nor ${MAIN_COMMIT:0:7}"
      return
    }
    why+=("installed executables are f930fbc")
  fi
  checkout_at_main || why+=("checkout services/ is not ${MAIN_COMMIT:0:7}")
  if ((${#why[@]})); then echo "pending: $(join "${why[@]}")"; else echo "done"; fi
}

state_register() {
  case $(helper register-state "$REGISTRY" "$PROVIDER" "$MANIFEST_MD5") in
  match) echo "done" ;;
  absent) echo "pending: $PROVIDER is not registered" ;;
  *) echo "blocked: $PROVIDER is registered with another manifest; a corrected" \
    "registration needs a new provider ID (HADES_MW_PROVIDER)" ;;
  esac
}

state_configure() {
  local dir why=() yaml_state
  for dir in "$LEDGER" "$RUN_DIR"; do
    if [[ -e $dir ]]; then
      private_dir "$dir" || {
        echo "blocked: $dir is not a 0700 directory owned by this account"
        return
      }
    else
      why+=("$(basename "$dir") missing")
    fi
  done
  [[ -f $YAML ]] || {
    echo "blocked: the HADES YAML $YAML does not exist"
    return
  }
  yaml_state=$(helper text-state "$YAML" <(render_yaml_block) registered_providers setup)
  case $yaml_state in
  match) ;;
  absent) why+=("YAML sections absent") ;;
  *)
    echo "blocked: the YAML has registered_providers or setup with other values;" \
      "edit it by hand"
    return
    ;;
  esac
  if [[ -e $VERIFIED_ENV ]]; then
    cmp -s <(render_verified_env) "$VERIFIED_ENV" || {
      echo "blocked: $VERIFIED_ENV exists with other content"
      return
    }
  else
    why+=("embedder-verified.env missing")
  fi
  if ((${#why[@]})); then echo "pending: $(join "${why[@]}")"; else echo "done"; fi
}

state_switch_embedder() {
  [[ -f $TEMPLATE ]] || {
    echo "blocked: the unit template is missing from this checkout"
    return
  }
  if [[ -e $UNIT ]]; then
    cmp -s <(render_unit) "$UNIT" || {
      echo "blocked: the installed verified unit differs from the template; review it"
      return
    }
  else
    echo "pending: verified unit not installed"
    return
  fi
  unit_enabled hades-embedder.service && {
    echo "pending: legacy unit still enabled"
    return
  }
  if unit_enabled hades-embedder-verified.service &&
    unit_active hades-embedder-verified.service; then
    echo "done"
  else
    echo "pending: verified unit not enabled and active"
  fi
}

inspect_state() {
  helper receipt-state <("$HADES" db setup inspect --operation-id "$OPERATION_ID" \
    2>/dev/null || true) "$OPERATION_ID" "$DB"
}

state_setup_db() {
  [[ $(state_configure) == "done" ]] || {
    echo "pending: configure first"
    return
  }
  "$HADES" db setup --help >/dev/null 2>&1 || {
    echo "pending: the installed hades has no db setup"
    return
  }
  local receipt
  receipt=$(inspect_state) || {
    echo "blocked: the ledger holds $OPERATION_ID for another database"
    return
  }
  case $receipt in
  ready) echo "done" ;;
  none) echo "pending: no receipt for $OPERATION_ID" ;;
  error) echo "blocked: db setup inspect failed; run it by hand" ;;
  *) echo "pending: receipt is $receipt; setup-db consults diagnose" ;;
  esac
}

backup_state() {
  local name=$1 dir
  [[ -f $STATE/$name.path ]] || {
    echo "pending: no $name recorded"
    return
  }
  dir=$(<"$STATE/$name.path")
  if [[ -f $dir/manifest.json ]] && grep -q '"managed_workspace"' "$dir/manifest.json"; then
    echo "done"
  else
    echo "blocked: the recorded $name has no managed manifest; inspect $dir"
  fi
}

state_backup_1() { backup_state backup-1; }
state_backup_2() { backup_state backup-2; }

state_move_desk() {
  [[ -f $STATE/move-desk.done ]] && {
    echo "done"
    return
  }
  local phases=()
  for phase in final.path final-verified restored switched; do
    [[ -f $STATE/move-desk/$phase ]] && phases+=("$phase")
  done
  if ((${#phases[@]})); then
    echo "pending: resumes after ${phases[*]}"
  else
    echo "pending: not started"
  fi
}

state_ingest() {
  if [[ -f $STATE/ingest.done ]]; then echo "done"; else echo "pending: not ingested"; fi
}

state_of() {
  local fn=state_${1//-/_}
  "$fn"
}

# Preconditions shared by several steps -------------------------------------------------

require_done() {
  local step
  for step in "$@"; do
    [[ $(state_of "$step") == "done" ]] || refuse "$step is not done; run it first"
  done
}

# The driver writes the manifest the verifier reads. It is the operator's to extend;
# a backup of the managed database needs it to take the database by name.
require_driver() {
  [[ -f $DRIVER ]] || refuse "the private backup driver is missing"
  grep -q -- '--database' "$DRIVER" ||
    refuse "the backup driver does not take --database yet; extend it first" \
      "(docs/operations/thinkpad-backup-restore.md, Driver changes)"
}

require_tmp() {
  local need=$1 have
  have=$(avail_gib /tmp)
  ((have >= need)) || refuse "/tmp has $have GiB available; this needs $need"
}

# Run the driver for one database; print the new backup directory it recorded.
take_backup() {
  local database=$1 before=""
  [[ -f $LATEST ]] && before=$(<"$LATEST")
  run python3 "$DRIVER" --database "$database"
  if ((DRY_RUN)); then
    BACKUP_DIR="<new-backup-dir>"
    return 0
  fi
  BACKUP_DIR=$(<"$LATEST")
  [[ $BACKUP_DIR != "$before" && -f $BACKUP_DIR/manifest.json ]] ||
    refuse "the driver did not record a new backup"
}

# Verify a backup in isolation, keep result.json, and remove the verifier's staging.
verify_backup() {
  local backup=$1 name=$2 dir
  run_tee "$RECORD/$name-verify.log" python3 "$REPO/scripts/verify_workspace_backup.py" \
    --backup "$backup" --arangod "$ARANGO_HOME/sbin/arangod" \
    --arangorestore "$ARANGO_HOME/bin/arangorestore" --hades "$HADES"
  if ((DRY_RUN)); then
    dir="/tmp/hades-recovery-<printed>"
  else
    dir=$(helper verifier-dir "$RECORD/$name-verify.log")
  fi
  run cp "$dir/result.json" "$RECORD/$name-verify-result.json"
  run rm -rf -- "$dir"
}

combined_backup() {
  local name=$1
  take_backup "$DB"
  run_to "$RECORD/$name-managed.json" helper add-managed "$BACKUP_DIR" "$DB" "$LEDGER" \
    "$REGISTRY"
  run helper check-manifest "$BACKUP_DIR" "$DB"
  run cp "$BACKUP_DIR/manifest.json" "$RECORD/$name-manifest.json"
}

# Steps ---------------------------------------------------------------------------------

act_install_main() {
  if ! installed_build; then
    check_hashes "$HADES_SHA256" "$BUILD/hades" "$VIEWER_SHA256" "$BUILD/hades-viewer"
    [[ -f $LOCAL_INSTALL/preflight-224.py ]] || refuse "the private preflight is missing"
    need_env
    confirm "Stop the viewer and the daemon and install ${MAIN_COMMIT:0:7}?"
    local backup snapshot
    backup=$DATA_LINK/backups/main-${MAIN_COMMIT:0:7}-$(date -u +%Y%m%dT%H%M%SZ)
    snapshot=(python3 -c 'import json, runpy, sys
s = runpy.run_path(sys.argv[1])["snapshot"]()
print(json.dumps(s, sort_keys=True))' "$LOCAL_INSTALL/preflight-224.py")
    run install -d -m 700 "$backup"
    run cp -p "$BIN_DIR/hades" "$BIN_DIR/hades-viewer" "$YAML" "$backup/"
    run_to "$backup/before.json" "${snapshot[@]}"
    run python3 "$LOCAL_INSTALL/preflight-224.py"
    run systemctl --user stop hades-viewer.service
    run systemctl --user stop hades-daemon.service
    run install -m 755 "$BUILD/hades" "$BIN_DIR/.hades.new"
    run install -m 755 "$BUILD/hades-viewer" "$BIN_DIR/.hades-viewer.new"
    run mv -f "$BIN_DIR/.hades.new" "$BIN_DIR/hades"
    run mv -f "$BIN_DIR/.hades-viewer.new" "$BIN_DIR/hades-viewer"
    run systemctl --user daemon-reload
    run systemctl --user start hades-daemon.service
    run python3 "$REPO/scripts/check_daemon_ready.py" --socket "$DAEMON_SOCK" \
      --database "$SOURCE_DB" --timeout 60
    run systemctl --user start hades-viewer.service
    # The post-install checks of the 52d04d4 procedure, against the new hashes.
    run_to "$RECORD/install-sha256.txt" sha256sum "$BIN_DIR/hades" "$BIN_DIR/hades-viewer"
    check_hashes "$HADES_SHA256" "$BIN_DIR/hades" "$VIEWER_SHA256" "$BIN_DIR/hades-viewer"
    local unit pid
    for unit in hades-daemon hades-viewer; do
      if ((DRY_RUN)); then
        show readlink "/proc/<MainPID-of-$unit>/exe"
      else
        pid=$(systemctl --user show "$unit.service" -p MainPID --value)
        readlink "/proc/$pid/exe" | tee -a "$RECORD/install-exe.txt"
        grep -q '(deleted)' "$RECORD/install-exe.txt" && refuse "$unit runs a deleted executable"
      fi
    done
    run systemctl --user is-active hades-arangodb hades-extractor hades-embedder \
      hades-daemon hades-viewer
    run python3 "$LOCAL_INSTALL/check-installed-services.py"
    run node "$LOCAL_INSTALL/check-installed-reading-desk.cjs"
    run node "$LOCAL_INSTALL/check-installed-workspace-chat.cjs"
    run_to "$backup/after.json" "${snapshot[@]}"
    run cmp "$backup/before.json" "$backup/after.json"
    say "installed; to roll back, see docs/installation-thinkpad.md, Roll back 52d04d4," \
      "with $backup"
  fi
  if ! checkout_at_main; then
    if ((!DRY_RUN)) && [[ -n $(git -C "$CHECKOUT" status --short -- services scripts) ]]; then
      refuse "the checkout has local changes under services/ or scripts/"
    fi
    run git -C "$CHECKOUT" fetch origin
    run git -C "$CHECKOUT" merge --ff-only "$MAIN_COMMIT"
    if ((!DRY_RUN)); then
      checkout_at_main || refuse "the checkout's services/ still differs from $MAIN_COMMIT"
    fi
  fi
}

act_register() {
  [[ -d $SNAPSHOT ]] || refuse "the approved snapshot $SNAPSHOT is missing"
  [[ -x $CHECKOUT/services/.venv/bin/python ]] || refuse "the services venv is missing"
  [[ -f $CHECKOUT/services/embedding/registry.py ]] ||
    refuse "the checkout's services/ has no embedding.registry; run install-main"
  # Registration copies 7.4 GiB of runtime files and freezes them read-only.
  local have
  have=$(avail_gib "$DATA")
  ((have >= 10)) || refuse "$DATA has $have GiB available; registration needs 10"
  run install -d -m 700 "$REGISTRY"
  run_to "$RECORD/register.json" env -C "$CHECKOUT/services" .venv/bin/python \
    -m embedding.registry --registry "$REGISTRY" register --provider "$PROVIDER" \
    --source "$SNAPSHOT" --revision "$REVISION"
  run helper check-register "$RECORD/register.json" "$MANIFEST_MD5"
}

act_configure() {
  [[ $(state_register) == "done" ]] || refuse "register is not done; run it first"
  "$HADES" db setup --help >/dev/null 2>&1 ||
    refuse "the installed hades has no db setup; run install-main"
  local dir
  for dir in "$LEDGER" "$RUN_DIR"; do
    [[ -e $dir ]] || run install -d -m 700 "$dir"
  done
  run stat -c '%a %U %n' "$REGISTRY" "$LEDGER" "$RUN_DIR"
  if [[ $(helper text-state "$YAML" <(render_yaml_block) registered_providers \
    setup) == absent ]]; then
    run cp -p "$YAML" "$RECORD/hades.yaml.before-configure"
    append_file "$YAML" render_yaml_block
  fi
  # A read-only verb must parse the configuration and leave the ledger untouched. On an
  # empty ledger it exits 1, names the missing operation on stderr, and prints a null
  # receipt; that is the expected answer here.
  run_to_status "$RECORD/configure-inspect.json" "$HADES" db setup inspect \
    --operation-id "$OPERATION_ID"
  if ((!DRY_RUN)); then
    [[ $(helper receipt-state "$RECORD/configure-inspect.json" "$OPERATION_ID" "$DB") == \
      none ]] || refuse "inspect did not report an empty ledger; see configure-inspect.json"
    [[ -z $(ls -A "$LEDGER") ]] || refuse "the ledger is not empty after a read-only inspect"
  fi
  [[ -e $VERIFIED_ENV ]] || write_file 600 "$VERIFIED_ENV" render_verified_env
}

print_switch_back() {
  say "To switch back to the legacy embedder (this stops managed ingest and search):"
  say "  systemctl --user disable --now hades-embedder-verified.service"
  say "  systemctl --user enable --now hades-embedder.service"
  say "Then read: journalctl --user -u hades-embedder-verified.service -n 50"
}

act_switch_embedder() {
  [[ $(state_configure) == "done" ]] || refuse "configure is not done; run it first"
  confirm "Disable the legacy embedder and start the verified provider unit?"
  if [[ ! -e $UNIT ]]; then
    write_file 644 "$UNIT" render_unit
    if ((!DRY_RUN)) && grep -n '@[A-Z_]*@' "$UNIT"; then
      refuse "the installed unit still holds a placeholder"
    fi
    run grep -nE '^(WorkingDirectory|ExecStart)|--registry' "$UNIT"
    run systemctl --user daemon-reload
  fi
  # Disabled, not only stopped, so a reboot cannot start two loads of the model.
  run systemctl --user disable --now hades-embedder.service
  run systemctl --user enable --now hades-embedder-verified.service
  # The model loads before the socket binds: about 20 s in the qualification.
  STATUS=0
  run_to_status "$RECORD/status.json" curl -sS --retry 60 --retry-all-errors \
    --retry-delay 2 --unix-socket "$SOCK" http://localhost/v1/status
  if ((!DRY_RUN)); then
    journalctl --user -u hades-embedder-verified.service -b --no-pager -n 80 \
      >"$RECORD/unit-first-start.journal" || true
    if ((STATUS)) || ! helper check-status "$RECORD/status.json" "$MODEL_ID" late; then
      print_switch_back
      refuse "the verified provider did not report a loaded model; journal in the record"
    fi
  fi
  run_to "$RECORD/hades-status.json" "$HADES" status
  run_to "$RECORD/verify.json" curl -sS --unix-socket "$SOCK" -X POST \
    http://localhost/v1/verify
  run helper check-verify "$RECORD/verify.json" "$REVISION" "$MANIFEST_MD5" \
    "$MAX_INPUT_TOKENS"
  say "While the verified unit holds the card, legacy embedding paths fail with a" \
    "connection error: hades embed, codebase ingest, ingest into a legacy database," \
    "and semantic search of a legacy database."
}

act_setup_db() {
  require_done switch-embedder
  need_env
  if ((!DRY_RUN)); then
    helper check-status <(curl -sS --unix-socket "$SOCK" http://localhost/v1/status) \
      "$MODEL_ID" late >/dev/null || refuse "the verified provider is not serving"
    local receipt action
    receipt=$(inspect_state) || refuse "the ledger holds $OPERATION_ID for another database"
    if [[ $receipt != none ]]; then
      action=$(helper diagnose-action <("$HADES" db setup diagnose \
        --operation-id "$OPERATION_ID"))
      [[ $action == retry_identical_request ]] ||
        refuse "receipt is $receipt and diagnose recommends $action; keep the ledger" \
          "and database untouched and inspect them"
      say "diagnose recommends retry_identical_request; repeating the exact request"
    fi
  fi
  run_to_status "$RECORD/setup-run.json" "$HADES" db setup run --database "$DB" \
    --provider "$PROVIDER" --operation-id "$OPERATION_ID" --generation "$GENERATION" \
    --chunking late --context-cap-tokens "$CONTEXT_CAP_TOKENS" \
    --window-overlap-tokens "$WINDOW_OVERLAP_TOKENS" --passage-tokens "$PASSAGE_TOKENS" \
    --overlap-tokens "$OVERLAP_TOKENS"
  run_to_status "$RECORD/setup-inspect.json" "$HADES" db setup inspect \
    --operation-id "$OPERATION_ID"
  run_to_status "$RECORD/setup-diagnose.json" "$HADES" db setup diagnose \
    --operation-id "$OPERATION_ID"
  run helper check-receipt "$RECORD/setup-run.json" "$RECORD/setup-inspect.json" \
    "$RECORD/setup-diagnose.json" "$OPERATION_ID" "$DB"
}

act_backup_1() {
  require_done setup-db
  [[ -f $STATE/move-desk.done || -d $STATE/move-desk || -f $STATE/ingest.done ]] &&
    refuse "backup-1 must come before the desk move and the first ingest"
  require_driver
  need_env
  combined_backup backup-1
  # The verifier needs the reading-desk collections, which arrive in move-desk;
  # backup-2 is the one verified in isolation.
  mark backup-1.path "$BACKUP_DIR"
}

aql() {
  local database=$1 out=$2
  shift 2
  run_to "$out" "$HADES" --db "$database" db aql -f json "$@"
}

counts() {
  local bind="<the names in moving.txt as {\"names\": [...]}>"
  ((DRY_RUN)) || bind=$(helper names-bind "$RECORD/moving.txt")
  aql "$1" "$2" -b "$bind" 'FOR n IN @names RETURN [n, COLLECTION_COUNT(n)]'
}

graphs() {
  aql "$1" "$2" 'FOR g IN _graphs SORT g._key RETURN UNSET(g, "_id", "_rev")'
}

# Compare the moving collections' counts and the graph definitions of both databases.
compare_move() {
  counts "$SOURCE_DB" "$RECORD/counts-before.raw.json"
  counts "$DB" "$RECORD/counts-after.raw.json"
  graphs "$SOURCE_DB" "$RECORD/graphs-before.raw.json"
  graphs "$DB" "$RECORD/graphs-after.raw.json"
  local side
  for side in counts-before counts-after graphs-before graphs-after; do
    run_to "$RECORD/$side.json" helper results "$RECORD/$side.raw.json"
  done
  ((DRY_RUN)) && return 0
  [[ -s $RECORD/counts-before.json && -s $RECORD/graphs-before.json ]] ||
    refuse "the source comparison came back empty"
  cmp -s "$RECORD/counts-before.json" "$RECORD/counts-after.json" &&
    cmp -s "$RECORD/graphs-before.json" "$RECORD/graphs-after.json"
}

desk() {
  local database=$1 out=$2 view
  ((DRY_RUN)) || : >"$out"
  for view in state archived sources; do
    run_to "$RECORD/desk-$database-$view.json" "$HADES" --db "$database" -f json reading \
      read --view "$view"
    if ((!DRY_RUN)); then
      helper desk "$RECORD/desk-$database-$view.json" >>"$out"
    fi
  done
}

restore() {
  # The password reaches arangorestore through its @VAR@ expansion, never argv, and
  # only this subshell's environment holds it.
  (
    export HADES_RESTORE_PASSWORD
    HADES_RESTORE_PASSWORD=$(<"$ROOT_PASSWORD_FILE")
    run "$ARANGO_HOME/bin/arangorestore" --configuration none \
      --server.endpoint "unix://$ARANGO_SOCK" --server.database "$DB" \
      --server.username root --server.password @HADES_RESTORE_PASSWORD@ \
      --create-database false --input-directory "$1" "${@:2}"
  )
}

services_stopped_notice() {
  say "hades-daemon and hades-viewer may be stopped. Re-run move-desk to resume, or undo" \
    "as docs/installation-thinkpad.md describes."
}

act_move_desk() {
  require_done backup-1
  require_driver
  need_env
  [[ -f $ROOT_PASSWORD_FILE ]] || refuse "the ArangoDB root password file is missing"
  [[ -x $ARANGO_HOME/bin/arangorestore ]] || refuse "arangorestore is missing"
  confirm "Back up $SOURCE_DB, stop the daemon and the viewer, and move the desk to $DB?"
  local phase_dir=$STATE/move-desk backup
  if ((!DRY_RUN)); then
    install -d -m 700 "$phase_dir"
    trap services_stopped_notice EXIT
  fi

  # A: the final source backup, the last before retirement. The driver restarts the
  # services when it finishes, so they are stopped again straight away.
  if [[ -f $phase_dir/final.path ]]; then
    backup=$(<"$phase_dir/final.path")
    say "final $SOURCE_DB backup already taken: $backup"
  else
    take_backup "$SOURCE_DB"
    backup=$BACKUP_DIR
    if ((!DRY_RUN)); then
      [[ $(helper json-field "$backup/dump/dump.json" database) == "$SOURCE_DB" ]] ||
        refuse "the final backup is not of $SOURCE_DB"
    fi
    mark move-desk/final.path "$backup"
  fi
  run systemctl --user stop hades-viewer.service
  run systemctl --user stop hades-daemon.service

  # B: what moves, from the quiet database and from the dump; they must agree.
  aql "$SOURCE_DB" "$RECORD/collections.json" 'RETURN COLLECTIONS()[*].name'
  run_to "$RECORD/moving.txt" helper moving-aql "$RECORD/collections.json"
  run_to "$RECORD/moving-dump.txt" helper moving-dump "$backup/dump"
  if ((!DRY_RUN)); then
    [[ -s $RECORD/moving.txt ]] || refuse "no reading collections to move"
    cmp "$RECORD/moving.txt" "$RECORD/moving-dump.txt" ||
      refuse "the database and the dump disagree on what moves; take the backup again"
  fi
  run helper staying "$RECORD/collections.json" "$RECORD/moving.txt"
  aql "$SOURCE_DB" "$RECORD/graphs-source.json" 'FOR g IN _graphs RETURN g'
  run helper check-graphs "$RECORD/graphs-source.json" "$RECORD/moving.txt"

  # C: verify the final backup in isolation and keep it.
  if [[ ! -f $phase_dir/final-verified ]]; then
    ((DRY_RUN)) || require_tmp 2
    verify_backup "$backup" conveyance-kg-final
    mark move-desk/final-verified "$RECORD"
  fi

  # D: restore from a staged copy, because arangorestore writes into its input
  # directory and the backup must keep exactly its manifested files.
  if [[ ! -f $phase_dir/restored ]]; then
    aql "$DB" "$RECORD/target-collections.json" 'RETURN COLLECTIONS()[*].name'
    aql "$DB" "$RECORD/target-graphs.json" 'RETURN LENGTH(_graphs)'
    if ((!DRY_RUN)); then
      if [[ -n $(helper moving-aql "$RECORD/target-collections.json") ]]; then
        compare_move || refuse "$DB already holds reading collections that differ from" \
          "$SOURCE_DB; inspect by hand"
        say "$DB already holds an identical copy; not restoring again"
      else
        [[ $(helper results "$RECORD/target-graphs.json") == "[0]" ]] ||
          refuse "$DB already holds graph definitions"
      fi
    fi
    if ((DRY_RUN)) || [[ -z $(helper moving-aql "$RECORD/target-collections.json") ]]; then
      local stage args=()
      if ((DRY_RUN)); then
        stage="$RECORD/move-dump-XXXXXX"
        args=(--collection "<each-name-in-moving.txt>")
      else
        stage=$(mktemp -d "$RECORD/move-dump-XXXXXX")
        local name names
        mapfile -t names <"$RECORD/moving.txt"
        for name in "${names[@]}"; do args+=(--collection "$name"); done
      fi
      run cp -a "$backup/dump/." "$stage/"
      # A name clash fails rather than replacing a collection.
      restore "$stage" --overwrite false "${args[@]}"
      # Graph definitions into the existing, empty _graphs, without recreating it.
      restore "$stage" --include-system-collections true --create-collection false \
        --collection _graphs
      run rm -rf -- "$stage"
      if ((!DRY_RUN)); then
        compare_move || refuse "counts or graph definitions differ after the restore"
      else
        compare_move
      fi
    fi
    run_to_status "$RECORD/setup-diagnose-moved.json" "$HADES" db setup diagnose \
      --operation-id "$OPERATION_ID"
    if ((!DRY_RUN)); then
      [[ $(helper diagnose-action "$RECORD/setup-diagnose-moved.json") == none ]] ||
        refuse "diagnose no longer reports action none after the move"
    fi
    mark move-desk/restored "$RECORD"
  fi

  # E: point the daemon and the viewer at the new database (drop-ins and daemon.env).
  if [[ ! -f $phase_dir/switched ]]; then
    local dropin generator
    for dropin in "$DAEMON_DROPIN" "$VIEWER_DROPIN"; do
      generator=render_daemon_dropin
      [[ $dropin == "$VIEWER_DROPIN" ]] && generator=render_viewer_dropin
      if [[ -e $dropin ]]; then
        cmp -s <("$generator") "$dropin" || refuse "$dropin exists with other content"
      else
        run install -d -m 755 "$(dirname "$dropin")"
        write_file 644 "$dropin" "$generator"
      fi
    done
    if ((!DRY_RUN)); then
      if ! grep -Eq -- "--db $DB .*--mcp-dbs $DB " "$DAEMON_DROPIN" ||
        grep -Eq -- "--(db|mcp-dbs) $SOURCE_DB " "$DAEMON_DROPIN"; then
        refuse "the daemon drop-in does not name $DB"
      fi
      ! grep -Eq "\b$SOURCE_DB\b" "$VIEWER_DROPIN" ||
        refuse "the viewer drop-in still names $SOURCE_DB"
    fi
    if ! grep -qx "HADES_DATABASE=$DB" "$DAEMON_ENV"; then
      grep -qx "HADES_DATABASE=$SOURCE_DB" "$DAEMON_ENV" ||
        refuse "daemon.env names neither $SOURCE_DB nor $DB"
      run cp -p "$DAEMON_ENV" "$RECORD/daemon.env.before-move-desk"
      run sed -i "s/^HADES_DATABASE=$SOURCE_DB\$/HADES_DATABASE=$DB/" "$DAEMON_ENV"
    fi
    run systemctl --user daemon-reload
    run_to "$RECORD/units-after.txt" systemctl --user cat hades-daemon.service \
      hades-viewer.service
    mark move-desk/switched "$RECORD"
  fi
  run systemctl --user start hades-daemon.service
  run python3 "$REPO/scripts/check_daemon_ready.py" --socket "$DAEMON_SOCK" \
    --database "$DB" --timeout 60
  run systemctl --user start hades-viewer.service
  ((DRY_RUN)) || trap - EXIT

  # F: the desk check. The pin proves the graph definitions resolve, since the desk
  # checks the definition before every write; it is made by hand once the lists match.
  desk "$SOURCE_DB" "$RECORD/desk-before.json"
  desk "$DB" "$RECORD/desk-after.json"
  if ((!DRY_RUN)); then
    [[ -s $RECORD/desk-before.json ]] || refuse "the source desk lists came back empty"
    cmp "$RECORD/desk-before.json" "$RECORD/desk-after.json" ||
      refuse "the desk lists differ; do not pin until they match"
  fi
  mark move-desk.done "$RECORD"
  say "Now open http://127.0.0.1:8088/reading, confirm the same readings are listed," \
    "open one reading and make one pin. $SOURCE_DB keeps its copy unchanged."
}

corpus_files() {
  find "$CORPUS" -type f \( -name '*.md' -o -name '*.txt' \) | LC_ALL=C sort
}

act_ingest() {
  require_done move-desk
  need_env
  [[ -d $CORPUS ]] || refuse "the corpus $CORPUS is missing"
  [[ -z $(git -C "$CORPUS" status --short -- .) ]] ||
    refuse "the corpus tree has uncommitted changes"
  local corpus
  mapfile -t corpus < <(corpus_files)
  ((${#corpus[@]})) || refuse "the corpus has no Markdown or text files"
  say "${#corpus[@]} files under $CORPUS (Markdown and text only: a tree ingest would" \
    "also send JSON evidence and PNG screenshots to the extractor)"
  run_to "$RECORD/corpus-commit.txt" git -C "$CORPUS" rev-parse HEAD
  if ((DRY_RUN)); then
    say "+ list the files into $RECORD/corpus-files.txt; first and last:"
    say "    ${corpus[0]}"
    say "    ${corpus[-1]}"
  else
    printf '%s\n' "${corpus[@]}" >"$RECORD/corpus-files.txt"
  fi
  # --root keys each document by its path under the corpus, so README.md files differ.
  run_to_status "$RECORD/ingest.json" "$HADES" --db "$DB" ingest --root "$CORPUS" \
    "${corpus[@]}"
  run helper check-ingest "$RECORD/ingest.json" "${#corpus[@]}"
  mark ingest.done "$RECORD"
  run_to "$RECORD/query-inventory.json" "$HADES" --db "$DB" db query "$QUERY_INVENTORY" -n 5
  run_to "$RECORD/query-ceiling.json" "$HADES" --db "$DB" db query "$QUERY_CEILING" -n 5
  local failed=0
  run helper query-sources "$RECORD/query-inventory.json" "${EXPECT_INVENTORY[@]}" ||
    failed=1
  run helper query-sources "$RECORD/query-ceiling.json" "${EXPECT_CEILING[@]}" || failed=1
  ((!failed)) || refuse "ingest is recorded, but a question missed its expected sources;" \
    "see the query files in the record"
}

act_backup_2() {
  require_done ingest
  require_driver
  need_env
  # The verifier stages every manifested file under /tmp, registered artifacts
  # (7.4 GiB) included, and /tmp is a tmpfs shared with other work.
  ((DRY_RUN)) || require_tmp 15
  combined_backup backup-2
  verify_backup "$BACKUP_DIR" backup-2
  mark backup-2.path "$BACKUP_DIR"
}

retirement_gate() {
  cat <<EOF

Retirement gate. A verified backup recovers the records of $DB, not a ready managed
workspace: a restored database has a new incarnation that the copied setup receipt
cannot authorize, and ownership rebinding is not supported
(docs/operations/model-bound-recovery.md). $SOURCE_DB is retired only after the
operator records acceptance of that limitation, with the date, in
docs/installation-thinkpad.md, "Retirement gate". The drop is by hand, outside HADES.
This script never drops a database.
EOF
}

do_step() {
  local step=$1 state fn
  CURRENT=$step
  state=$(state_of "$step")
  case $state in
  done)
    say "$step: already done"
    return 0
    ;;
  blocked:*)
    refuse "${state#blocked: }"
    return 0
    ;;
  esac
  say "== $step (${state#pending: })"
  begin_record
  fn=act_${step//-/_}
  "$fn"
  ((DRY_RUN)) && return 0
  state=$(state_of "$step")
  [[ $state == "done" ]] || refuse "did not reach its postcondition: $state"
  say "$step: done; evidence in $RECORD"
}

# check ---------------------------------------------------------------------------------

item() {
  local label=$1
  shift
  if "$@"; then say "  ok   $label"; else say "  FAIL $label"; fi
}

build_hashes_ok() {
  sha_is "$BUILD/hades" "$HADES_SHA256" && sha_is "$BUILD/hades-viewer" "$VIEWER_SHA256"
}

arango_tools_ok() {
  [[ -x $ARANGO_HOME/bin/arangorestore && -x $ARANGO_HOME/sbin/arangod ]]
}

driver_ready() { [[ -f $DRIVER ]] && grep -q -- '--database' "$DRIVER"; }

corpus_clean() {
  [[ -d $CORPUS && -z $(git -C "$CORPUS" status --short -- . 2>/dev/null) ]]
}

check() {
  CURRENT=check
  say "Values"
  say "  database           $DB (source $SOURCE_DB)"
  say "  provider           $PROVIDER, $MODEL_ID at $REVISION, manifest MD5 $MANIFEST_MD5"
  say "  setup              installation $INSTALLATION_ID, operation $OPERATION_ID," \
    "generation $GENERATION"
  say "  late policy        context cap $CONTEXT_CAP_TOKENS, window overlap" \
    "$WINDOW_OVERLAP_TOKENS, passages $PASSAGE_TOKENS overlapping $OVERLAP_TOKENS," \
    "provider ceiling $MAX_INPUT_TOKENS"
  say "  main build         ${MAIN_COMMIT}, hades $HADES_SHA256"
  say "  paths              DATA=$(data_location) (resolved from $DATA_LINK); YAML $YAML;" \
    "checkout $CHECKOUT; corpus $CORPUS; records $RECORDS"
  say "Preconditions"
  item "daemon.env present" test -f "$DAEMON_ENV"
  item "HADES YAML present" test -f "$YAML"
  item "main build hashes match" build_hashes_ok
  item "approved snapshot present" test -d "$SNAPSHOT"
  item "services venv present" test -x "$CHECKOUT/services/.venv/bin/python"
  item "ArangoDB tools present" arango_tools_ok
  item "root password file present" test -f "$ROOT_PASSWORD_FILE"
  item "backup driver takes --database" driver_ready
  item "data directory has 10 GiB free ($(avail_gib "$DATA") GiB)" \
    test "$(avail_gib "$DATA")" -ge 10
  item "/tmp has 15 GiB free for backup-2 ($(avail_gib /tmp) GiB)" \
    test "$(avail_gib /tmp)" -ge 15
  item "corpus tree clean" corpus_clean
  say "Steps"
  local step
  for step in "${STEPS[@]}"; do
    say "  $(printf '%-16s' "$step") $(state_of "$step")"
  done
}

# Main ----------------------------------------------------------------------------------

usage() {
  sed -n '/^# Usage:/,/^# --dry-run/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

main() {
  local target=""
  while (($#)); do
    case $1 in
    --dry-run) DRY_RUN=1 ;;
    --yes) YES=1 ;;
    -h | --help) usage 0 ;;
    -*) usage 2 >&2 ;;
    *)
      [[ -z $target ]] || usage 2 >&2
      target=$1
      ;;
    esac
    shift
  done
  [[ -n $target ]] || usage 2 >&2
  [[ -d $DATA ]] || {
    echo "the data directory does not resolve to a directory" >&2
    exit 1
  }
  command -v python3 >/dev/null || {
    echo "python3 is required" >&2
    exit 1
  }
  load_env
  case $target in
  check) check ;;
  all)
    local step
    for step in "${STEPS[@]}"; do do_step "$step"; done
    retirement_gate
    ;;
  *)
    local known=" ${STEPS[*]} "
    [[ $known == *" $target "* ]] || usage 2 >&2
    do_step "$target"
    [[ $target != backup-2 ]] || retirement_gate
    ;;
  esac
}

main "$@"
