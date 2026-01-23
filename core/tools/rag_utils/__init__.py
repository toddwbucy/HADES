"""
RAG Utils - Universal Academic Tools
====================================

Source-agnostic utilities for building Retrieval-Augmented Generation (RAG)
systems from academic corpora. These tools work with any academic paper source:
ArXiv, SSRN, PubMed, Harvard Law Library, or any other collection.

Key Modules:
- academic_citation_toolkit: Universal citation and bibliography extraction
"""

from .academic_citation_toolkit import (
    ArangoCitationStorage,
    # Concrete implementations
    ArangoDocumentProvider,
    # Core data models
    BibliographyEntry,
    CitationStorage,
    # Abstract interfaces
    DocumentProvider,
    FileSystemDocumentProvider,
    InTextCitation,
    JSONCitationStorage,
    # Main processing classes
    UniversalBibliographyExtractor,
    UniversalCitationExtractor,
    # Factory functions
    create_arxiv_citation_toolkit,
    create_filesystem_citation_toolkit,
)

__version__ = "1.0.0"
__author__ = "Claude Code + HADES-Lab Team"

__all__ = [
    # Data models
    'BibliographyEntry',
    'InTextCitation',

    # Interfaces
    'DocumentProvider',
    'CitationStorage',

    # Providers
    'ArangoDocumentProvider',
    'FileSystemDocumentProvider',

    # Storage
    'ArangoCitationStorage',
    'JSONCitationStorage',

    # Extractors
    'UniversalBibliographyExtractor',
    'UniversalCitationExtractor',

    # Factory functions
    'create_arxiv_citation_toolkit',
    'create_filesystem_citation_toolkit',
]
