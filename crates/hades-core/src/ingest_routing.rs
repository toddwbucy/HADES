//! One routing table: what a file extension means for ingestion.
//!
//! The tree used to have two ingest commands, one for code and one for
//! documents, and the operator had to know which was which. That is how a
//! mixed tree lost half of itself: `codebase ingest` reported markdown as "no
//! handler for extension" and `hades ingest` sent a `.py` file to docling. The
//! extension is the only thing that decides, and it decides in one place.
//!
//! [`Route`] is deliberately exhaustive over the decision rather than over the
//! languages: a caller that adds an extension has to say which pipeline claims
//! it, and a file that matches nothing is reported rather than dropped.

use std::path::Path;

use crate::code::Language;

/// Which pipeline claims a file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Route {
    /// Source code with a dedicated or structural analyzer: symbols, edges,
    /// AST-aligned chunks, late-chunked embeddings.
    Code(Language),
    /// A document: extraction (docling, LaTeX, or a plain read), then chunking
    /// and embedding into the document profile.
    Document,
    /// Nothing claims it. The caller reports these rather than ignoring them,
    /// because "silently skipped" is the failure this module exists to end.
    Unrouted,
}

/// Extensions the extraction service can turn into text.
///
/// This list is what the service *accepts*, verified against it rather than
/// inferred from what docling can do in principle:
///
/// - `md`, `markdown`, `txt`, `text`, `rst` take the server's plain-read path
/// - `pdf` goes to docling, whose `SUPPORTED_EXTENSIONS` is `{pdf, txt, text, md}`
/// - `tex` and `gz` go to the LaTeX backend
///
/// **`html`, `htm`, `csv`, `docx`, `pptx` and `xlsx` were briefly listed here** on
/// the assumption that docling's unknown-format fallback would take them. It does
/// not: `docling_backend.py:122` returns "Unsupported format" for anything outside
/// that four-extension set, and the 13 `.html` templates in one real corpus failed
/// extraction after the walk had already claimed them. Claiming a file and failing
/// it is worse than declining it, because `unrouted` is a list an operator reads
/// while a failed document is a line in a summary.
///
/// Adding one back means teaching the extractor first and this table second.
const DOCUMENT_EXTENSIONS: &[&str] = &["md", "markdown", "txt", "text", "rst", "pdf", "tex", "gz"];

/// Route one path by its extension.
///
/// Code wins over documents where both could claim an extension, because a
/// dedicated analyzer produces symbols and edges that extraction cannot.
pub fn route_for(path: &Path) -> Route {
    if let Some(lang) = Language::from_path(&path.to_string_lossy()) {
        return Route::Code(lang);
    }
    let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
        return Route::Unrouted;
    };
    if DOCUMENT_EXTENSIONS.contains(&ext.to_ascii_lowercase().as_str()) {
        return Route::Document;
    }
    Route::Unrouted
}

/// Whether this extension is claimed by the document pipeline.
pub fn is_document_extension(ext: &str) -> bool {
    DOCUMENT_EXTENSIONS.contains(&ext.to_ascii_lowercase().as_str())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn code_extensions_route_to_their_analyzer() {
        for (file, lang) in [
            ("src/main.rs", Language::Rust),
            ("app.py", Language::Python),
            ("kernel.cu", Language::Cpp),
            ("header.hpp", Language::Cpp),
            ("server.go", Language::Go),
        ] {
            assert_eq!(route_for(&PathBuf::from(file)), Route::Code(lang), "{file}");
        }
    }

    #[test]
    fn document_extensions_route_to_extraction() {
        for file in [
            "README.md",
            "spec.markdown",
            "notes.txt",
            "guide.rst",
            "paper.pdf",
            "paper.tex",
            "arxiv.gz",
        ] {
            assert_eq!(route_for(&PathBuf::from(file)), Route::Document, "{file}");
        }
    }

    /// Formats the extraction service refuses must be declined by the router,
    /// not claimed and then failed. 13 `.html` templates in one corpus were
    /// routed to extraction and came back "Unsupported format", which turns a
    /// readable `unrouted` entry into a failed document in a summary.
    #[test]
    fn formats_the_extractor_refuses_are_not_claimed() {
        for file in [
            "templates/instrument.html",
            "page.htm",
            "table.csv",
            "report.docx",
            "deck.pptx",
            "book.xlsx",
        ] {
            assert_eq!(route_for(&PathBuf::from(file)), Route::Unrouted, "{file}");
        }
    }

    /// The case that cost a corpus: a markdown file in a code tree.
    #[test]
    fn markdown_in_a_code_tree_is_not_unrouted() {
        assert_eq!(
            route_for(&PathBuf::from("crates/weaver-spu/README.md")),
            Route::Document
        );
    }

    #[test]
    fn unknown_and_extensionless_are_reported_not_guessed() {
        assert_eq!(
            route_for(&PathBuf::from("vendor/sigma.min.js")),
            Route::Unrouted
        );
        assert_eq!(route_for(&PathBuf::from("Makefile")), Route::Unrouted);
        assert_eq!(route_for(&PathBuf::from("image.png")), Route::Unrouted);
    }

    /// `.ts` has no analyzer and no registered grammar today, so it must not
    /// quietly become a document: it would be embedded as prose with no symbols
    /// and nothing would say so.
    #[test]
    fn typescript_is_unrouted_rather_than_treated_as_prose() {
        assert_eq!(route_for(&PathBuf::from("app.ts")), Route::Unrouted);
    }
}
