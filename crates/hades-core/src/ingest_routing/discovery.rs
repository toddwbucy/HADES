//! Shared tree discovery for foreground ingestion and background admission.

use super::{Route, route_for};
use crate::code::Language;
use anyhow::{Context, Result};
use ignore::WalkBuilder;
use std::path::{Path, PathBuf};

/// Hard floor for directory exclusions — applied even when the project has
/// no `.gitignore`, no `.ignore`, and no `.hadesignore`. The `ignore` crate's
/// standard filters (gitignore + hidden-file skipping) already handle most
/// real repos; this list catches the unfortunate case of a flat directory
/// dropped onto disk without any ignore files.
pub const SKIP_DIRS: &[&str] = &[
    "__pycache__",
    "node_modules",
    "target",
    "venv",
    "dist",
    "build",
];

/// A file under the ingest root that discovery deliberately did not pick up.
///
/// Recorded rather than dropped so `codebase drift` can report a third bucket.
/// A file that is neither ingested nor reportable is a silent hole: the pair
/// (ingest, drift) otherwise reports a clean sweep over a partially-covered
/// tree, which is a false green rather than a visible gap (#183).
#[derive(Debug, Clone, serde::Serialize)]
pub struct UnhandledFile {
    pub path: String,
    pub reason: &'static str,
}

/// The first line of a file, if it is readable as UTF-8.
///
/// Used only for shebang sniffing, so a binary (invalid UTF-8) simply yields
/// `None` and stays out of discovery.
///
/// The read is capped: `read_line` alone allocates until the first newline, so
/// an extensionless blob with none near the start (a compiled `a.out`, a
/// checked-in artifact, a minified single-line bundle) would be pulled entirely
/// into memory during a directory walk that previously never opened it.
/// A shebang lives in the first handful of bytes or not at all.
fn first_line(path: &Path) -> Option<String> {
    use std::io::{BufRead, BufReader, Read};
    /// Longest plausible shebang line; anything beyond cannot be one.
    const SHEBANG_PROBE_BYTES: u64 = 256;
    let file = std::fs::File::open(path).ok()?;
    let mut line = String::new();
    BufReader::new(file.take(SHEBANG_PROBE_BYTES))
        .read_line(&mut line)
        .ok()?;
    Some(line)
}

/// The analyzer language implied by a file's shebang, plus whether it had one.
///
/// Only consulted for extensionless files: an extension is cheaper and more
/// reliable when present.
/// Only extensionless files are sniffed. An extension is cheaper and more
/// reliable when present, and — critically — the ingest loop applies the same
/// restriction, so admitting an extension-bearing script here would let it pass
/// discovery and then fail with "cannot detect language" instead of the
/// actionable "unsupported file type … use --language or --unparsed-ext".
pub fn shebang_of(path: &Path) -> Option<(bool, Option<Language>)> {
    if path.extension().is_some() {
        return None;
    }
    let line = first_line(path)?;
    let trimmed = line.trim_end();
    if !Language::is_shebang(trimmed) {
        return None;
    }
    Some((true, Language::from_shebang(trimmed)))
}

/// Every file under `root`, classified by which pipeline claims it.
///
/// One walk, one routing decision per file, so a unified ingest can report what
/// happened to everything rather than leaving a caller to infer it from two
/// separate commands' counts. Honors the same ignore rules as code discovery:
/// `.gitignore`, `.ignore`, `.hadesignore` and [`SKIP_DIRS`].
///
/// Extensionless files with a shebang are classified as code, matching
/// `discover_files_detailed`, which includes them so they are visible to ingest
/// and drift instead of disappearing (#183).
///
/// `unparsed_set` are extensions the operator asked to embed without a parser.
/// They count as code here because that is the phase which ingests them, and a
/// report that called them unrouted while the code phase was storing them would
/// be wrong in the direction operators trust.
pub fn discover_by_route(
    root: &Path,
    unparsed_set: &std::collections::HashSet<String>,
) -> Result<RouteDiscovery> {
    let mut found = RouteDiscovery::default();

    let unparsed = |p: &Path| {
        p.extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| unparsed_set.contains(&e.to_lowercase()))
    };

    if root.is_file() {
        match route_for(root) {
            Route::Code(_) => found.code.push(root.to_path_buf()),
            Route::Document => found.documents.push(root.to_path_buf()),
            Route::Unrouted => {
                if unparsed(root) || shebang_of(root).is_some() {
                    found.code.push(root.to_path_buf());
                } else {
                    found.unrouted.push(UnhandledFile {
                        path: root.to_string_lossy().to_string(),
                        reason: "no handler for extension",
                    });
                }
            }
        }
        return Ok(found);
    }

    let walker = WalkBuilder::new(root)
        .follow_links(false)
        .add_custom_ignore_filename(".hadesignore")
        .filter_entry(|entry| {
            if entry.file_type().map(|t| t.is_dir()).unwrap_or(false)
                && let Some(name) = entry.file_name().to_str()
            {
                return !SKIP_DIRS.contains(&name);
            }
            true
        })
        .build();

    for entry in walker {
        let entry = entry.context("error walking directory")?;
        if !entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
            continue;
        }
        let path = entry.path();
        match route_for(path) {
            Route::Code(_) => found.code.push(path.to_path_buf()),
            Route::Document => found.documents.push(path.to_path_buf()),
            Route::Unrouted => {
                if unparsed(path) || (path.extension().is_none() && shebang_of(path).is_some()) {
                    found.code.push(path.to_path_buf());
                } else {
                    found.unrouted.push(UnhandledFile {
                        path: path.to_string_lossy().to_string(),
                        reason: if path.extension().is_some() {
                            "no handler for extension"
                        } else {
                            "no extension and no shebang"
                        },
                    });
                }
            }
        }
    }

    found.code.sort();
    found.documents.sort();
    found.unrouted.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(found)
}

/// What one walk found, split by route.
#[derive(Default)]
pub struct RouteDiscovery {
    pub code: Vec<PathBuf>,
    pub documents: Vec<PathBuf>,
    pub unrouted: Vec<UnhandledFile>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_first_line_probe_is_bounded() {
        let dir = tempfile::tempdir().unwrap();
        // A newline-free blob must not be read whole during a directory walk.
        let blob = dir.path().join("bigblob");
        std::fs::write(&blob, "x".repeat(2 * 1024 * 1024)).unwrap();
        let line = first_line(&blob).unwrap();
        assert!(line.len() <= 256, "probe read {} bytes", line.len());
        // And it is correctly not treated as a script.
        assert!(shebang_of(&blob).is_none());
    }
}
