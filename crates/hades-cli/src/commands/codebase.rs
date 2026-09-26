//! `hades codebase` subcommands.

use std::path::PathBuf;

use clap::Subcommand;

#[derive(Debug, Subcommand)]
pub enum CodebaseCmd {
    /// Ingest source code into the knowledge graph.
    Ingest {
        /// Path to file or directory to ingest.
        path: PathBuf,

        /// Programming language override (auto-detected if omitted).
        #[arg(short = 'l', long)]
        language: Option<String>,

        /// Run in batch mode.
        #[arg(short = 'b', long)]
        batch: bool,

        /// Comma-separated extensions to embed without a parser (e.g.
        /// `wgsl,vert`). Files with these extensions are chunked by size and
        /// embedded as features — no symbol/edge extraction. Their file nodes
        /// are merged (existing fields preserved), not overwritten.
        #[arg(long = "unparsed-ext", value_delimiter = ',')]
        unparsed_ext: Vec<String>,

        /// Path to `compile_commands.json` (or its containing directory) for
        /// compiler-grade C/C++/CUDA include, define, standard, and target
        /// configuration. When omitted, source ancestors and `build/` are
        /// searched automatically.
        #[arg(long = "compile-commands")]
        compile_commands: Option<PathBuf>,

        /// Re-ingest each file even if its content hash is unchanged.
        /// This rebuilds symbols, chunks, and embeddings under the same file
        /// identity while retaining inbound authored edges. Ordinary ingestion
        /// already detects body, signature, and comment edits using the full
        /// content hash; --force is not required just because symbol names
        /// stayed the same. Use it to rebuild unchanged files after changes to
        /// analyzers or embedding configuration, or to refresh dependencies.
        ///
        /// Pass the ORIGINAL ingest root, not a narrower path. Keys are
        /// derived relative to the path given (a file bases at its parent), so
        /// re-ingesting a single file or subdirectory writes duplicate nodes
        /// under re-based keys, purges nothing, and repairs nothing.
        ///
        /// If a rebuild drops a symbol that another file points at, those
        /// inbound edges are reported as `dangling_inbound_edges` — not
        /// deleted, since each records a real dependency. Re-resolve them by
        /// re-running `codebase ingest --force <the same ingest root>` (plain
        /// re-ingest skips the dependents, whose own content hash did not
        /// change), or run `hades codebase prune-orphans` to drop them; until
        /// then `codebase validate` will flag them.
        ///
        /// This never permits an analyzer-fidelity downgrade by itself, so a
        /// file whose stored analysis came from a richer analyzer than the one
        /// available now is still skipped — pass `--allow-analysis-downgrade`
        /// as well to refresh it.
        ///
        /// Go may replace an older semantic tier when gopls is scheduled,
        /// the file belongs to a discoverable Go module, and incoming analysis
        /// is richer than raw text. Enrichment runs afterwards; failure can
        /// leave earlier file updates committed and makes the run fail unless
        /// analysis downgrade was explicitly accepted.
        #[arg(long = "force", alias = "no-skip")]
        force: bool,

        /// Permit a lower-fidelity analyzer to replace previously stored
        /// semantic artifacts. This is separate from `--force` so a temporary
        /// analyzer outage cannot silently degrade the graph. Also accepts reported
        /// semantic request failures; content refreshes while affected prior semantic edges are retained.
        #[arg(long = "allow-analysis-downgrade")]
        allow_analysis_downgrade: bool,
    },

    /// Update an existing code graph node.
    Update {
        /// Path to file or directory to update.
        path: PathBuf,
    },

    /// Show code ingestion statistics.
    Stats,

    /// Validate codebase graph invariants (ontology spec §11).
    Validate,

    /// Remove orphaned symbols, chunks, embeddings, and dangling edges.
    ///
    /// Sweeps child records whose owning file node is already gone. To retire a
    /// file node whose *source file* was deleted, use `codebase retire`.
    PruneOrphans {
        /// Report what would be deleted without modifying the graph.
        #[arg(long)]
        dry_run: bool,
    },

    /// Compare the graph against the source tree it describes (read-only).
    ///
    /// `codebase validate` checks only internal consistency and cannot see any
    /// of this.
    ///
    /// Buckets: `stale` (a file node with no counterpart under this root),
    /// `uningested` (a source file with no node), `changed` (a matched file
    /// whose content differs from what was ingested), and `unhandled` (files
    /// under the root ingest has no handler for, with a reason for each).
    ///
    /// `changed.unverifiable` counts matched files that could not be compared at
    /// all, because they were ingested before `content_hash` existed or are no
    /// longer readable as text.
    ///
    /// `clean` is true only when stale, uningested, changed and
    /// `changed.unverifiable` are all zero. `unhandled` does not gate it, since
    /// every repository contains files no analyzer handles.
    ///
    /// Both drift and incremental ingestion compare full content hashes.
    /// Body, signature, and comment edits therefore trigger ordinary re-ingest.
    /// Missing stored content hashes also trigger reprocessing, subject to the
    /// analyzer-fidelity guard. Unreadable files cannot be repaired by ingest
    /// until they become readable.
    ///
    /// Unchanged files are reprocessed when relationships are pending, or when
    /// an available embedder needs to backfill missing vectors according to
    /// stored chunk/embedding counts. These counts do not verify model identity
    /// or vector freshness. Use --force with the original ingest root for an
    /// explicit rebuild; it does not override the analyzer-fidelity guard.
    ///
    /// `stale` is NOT "the source file was deleted". It is every node with no
    /// counterpart under the root you passed. Nodes belonging to another
    /// ingest root are excluded and counted separately as `other_roots`, so a
    /// database holding several trees no longer reports one tree's nodes as
    /// stale for another (#192).
    ///
    /// The exception is nodes ingested before HADES recorded `ingest_root`.
    /// Those cannot be attributed either way, so they are still compared and the
    /// stale ones are listed separately as `stale.unattributed_keys`, held out
    /// of `stale.keys` so a `--full` pipe into `codebase retire` cannot delete
    /// them unreviewed.
    ///
    /// `other_roots` reports the roots as well as the count, because a root
    /// *under* this one is usually a mis-rooted ingest of this same tree rather
    /// than a second graph: `codebase ingest` on a single file bases its keys at
    /// that file's parent.
    ///
    /// Pass the same discovery flags used at ingest time, and the same root —
    /// keys are relative to the ingest root, so a wrong root reports near-total
    /// drift in both directions rather than a small honest number.
    Drift {
        /// Ingest root the graph was built from.
        path: PathBuf,

        /// Programming language override (must match the ingest invocation).
        #[arg(short = 'l', long)]
        language: Option<String>,

        /// Extensions ingested without a parser (must match the ingest
        /// invocation), e.g. `wgsl,vert`.
        #[arg(long = "unparsed-ext", value_delimiter = ',')]
        unparsed_ext: Vec<String>,

        /// List every key instead of truncating. Use this to feed
        /// `codebase retire --from -`.
        ///
        /// `stale.keys` holds only nodes attributed to this ingest root, so it
        /// is what `retire` should be fed. Keys that could not be attributed are
        /// held out, in `stale.unattributed_keys`, for review — `retire` deletes
        /// each target's node, chunks, embeddings, symbols and incident edges,
        /// and those keys predate the attribution that would prove they belong
        /// to this tree.
        ///
        /// Re-ingesting a root attributes every node whose file still exists. A
        /// node whose file is already gone is never rediscovered, so no re-ingest
        /// can attribute it; that residue is pre-attribution backlog and only a
        /// reviewed retire clears it.
        #[arg(long)]
        full: bool,
    },

    /// Retire graph nodes whose source files are gone (complement of --force).
    ///
    /// Removes each target's file node, chunks, embeddings, symbols, and every
    /// codebase edge incident on the file or its symbols. Edges in other
    /// collections (authored bridges such as conformance verdicts) are reported
    /// separately and need `--yes`, since they are irreplaceable if the target
    /// list is wrong.
    ///
    /// Targets are always explicit — use `codebase drift` to discover them.
    Retire {
        /// File node key to retire. Repeatable.
        #[arg(long = "file")]
        files: Vec<String>,

        /// Read newline-separated keys from a file (`-` for stdin).
        /// Blank lines and `#` comments are ignored.
        #[arg(long = "from")]
        from: Option<PathBuf>,

        /// Report what would be removed without modifying the graph.
        #[arg(long)]
        dry_run: bool,

        /// Confirm removal of edges outside the codebase collections.
        #[arg(short = 'y', long)]
        yes: bool,
    },
}
