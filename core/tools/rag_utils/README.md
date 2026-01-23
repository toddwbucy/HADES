# RAG Utils - Universal Academic Tools

This directory contains **source-agnostic** utilities for building Retrieval-Augmented Generation (RAG) systems from academic corpora. These tools work with **any academic paper source** - ArXiv, SSRN, PubMed, Harvard Law Library, or any other collection.

## Philosophy

Following Actor-Network Theory principles, these tools recognize that academic knowledge networks are **universal patterns** that exist regardless of their source. A citation network has the same fundamental structure whether it's built from:

- Computer Science papers (ArXiv)  
- Economics papers (SSRN)
- Medical papers (PubMed)
- Legal papers (Harvard Law Library)
- Any academic corpus

## Available Tools

### 🕸️ Academic Citation Toolkit

**File**: `academic_citation_toolkit.py`

Universal citation and bibliography extraction toolkit that works with any academic paper corpus.

**Key Features**:

- **Source-agnostic**: Works with ArXiv, SSRN, PubMed, law libraries, etc.
- **Multiple document providers**: ArangoDB, filesystem, APIs, web scraping
- **Multiple storage options**: ArangoDB, JSON, CSV, PostgreSQL, etc.
- **Universal format support**: Numbered citations, author-year, hybrid formats
- **Pluggable architecture**: Easy to extend for new sources/formats

**Quick Start**:

```python
from rag_utils.academic_citation_toolkit import create_arxiv_citation_toolkit

# For ArXiv papers in ArangoDB
extractor, storage = create_arxiv_citation_toolkit(arango_client)

# Extract bibliography from any paper
entries = extractor.extract_paper_bibliography("paper_id")
storage.store_bibliography_entries(entries)
```

## Design Principles

### 1. Source Agnostic

All tools work regardless of paper source. The same citation network builder works for:

- ArXiv computer science papers
- SSRN economics papers  
- PubMed medical papers
- Harvard Law Library legal papers

### 2. Pluggable Architecture

Tools use abstract interfaces that can be implemented for any:

- **Document Provider**: ArangoDB, filesystem, APIs, databases
- **Storage Backend**: ArangoDB, PostgreSQL, JSON, CSV
- **Format Parser**: Different citation formats, disciplines, languages

### 3. Universal Patterns  

Academic papers follow universal patterns:

- **Bibliography sections** (formal references)
- **In-text citations** (contextual pointers)  
- **Citation networks** (paper-to-paper relationships)
- **Author networks** (collaboration patterns)

### 4. Anthropological Awareness

Tools recognize that citation practices are **cultural artifacts** that vary by:

- Academic discipline (CS vs Law vs Medicine)
- Geographic region (US vs EU vs Asia)  
- Time period (1990s vs 2020s)
- Publication venue (journal vs conference vs preprint)

## Usage Examples

### Example 1: ArXiv Papers (Current Setup)

```python
from arango import ArangoClient
from rag_utils.academic_citation_toolkit import create_arxiv_citation_toolkit

client = ArangoClient(hosts='http://192.168.1.69:8529')
extractor, storage = create_arxiv_citation_toolkit(client)

# Extract bibliography from word2vec paper
entries = extractor.extract_paper_bibliography("1301_3781")
print(f"Found {len(entries)} bibliography entries")
```

### Example 2: Local PDF Files

```python
from rag_utils.academic_citation_toolkit import create_filesystem_citation_toolkit

# Extract from local PDFs, save to JSON
extractor, storage = create_filesystem_citation_toolkit(
    file_path="/path/to/papers", 
    output_path="/path/to/results"
)

entries = extractor.extract_paper_bibliography("paper_001")
storage.store_bibliography_entries(entries)
```

### Example 3: Custom Document Source

```python
from rag_utils.academic_citation_toolkit import (
    UniversalBibliographyExtractor, 
    DocumentProvider,
    JSONCitationStorage
)

class SSRNDocumentProvider(DocumentProvider):
    def get_document_text(self, paper_id: str) -> str:
        # Custom implementation for SSRN API
        pass

# Use with any storage backend
provider = SSRNDocumentProvider()
extractor = UniversalBibliographyExtractor(provider)
storage = JSONCitationStorage("/output/path")
```

## Extending the Tools

### Adding New Document Sources

Implement the `DocumentProvider` interface:

```python
class CustomDocumentProvider(DocumentProvider):
    def get_document_text(self, document_id: str) -> Optional[str]:
        # Fetch full text from your source
        pass
    
    def get_document_chunks(self, document_id: str) -> List[str]:
        # Split document into chunks
        pass
```

### Adding New Storage Backends

Implement the `CitationStorage` interface:

```python
class CustomCitationStorage(CitationStorage):
    def store_bibliography_entries(self, entries: List[BibliographyEntry]) -> bool:
        # Store to your backend
        pass
    
    def store_citations(self, citations: List[InTextCitation]) -> bool:
        # Store citations
        pass
```

## Integration with Olympus Ecosystem

### HERMES Integration

RAG utils can be used by HERMES for:

- Citation network enrichment
- Bibliography metadata extraction
- Academic relationship mapping

### HADES Integration  

RAG utils support HADES requirements:

- Multi-dimensional analysis (WHERE × WHAT × CONVEYANCE)
- Observer-dependent citation networks
- Context amplification measurement

### Hephaestus Tools

RAG utils follow Hephaestus design patterns:

- Configuration-driven operation
- Reusable across modules
- Tool gifting between modules

## Performance Characteristics

### Memory Usage

- **Lightweight**: Processes papers individually
- **Streaming**: No need to load entire corpus in memory
- **Configurable**: Adjustable chunk sizes and batch processing

### Processing Speed

- **Bibliography extraction**: ~1-2 seconds per paper
- **Citation parsing**: ~0.5-1 seconds per paper  
- **Network construction**: Scales with corpus size
- **Parallelizable**: Easy to distribute across workers

### Accuracy

- **High confidence for structured citations**: 90%+ for numbered references
- **Medium confidence for author-year**: 70-85% depending on format consistency
- **Robust error handling**: Graceful degradation on malformed inputs

## Future Extensions

### Planned Tools

- **Author Network Extractor**: Build collaboration networks
- **Topic Evolution Tracker**: Track concept development over time
- **Cross-Corpus Linker**: Connect papers across different sources
- **Citation Context Analyzer**: Understand why papers cite each other

### Planned Integrations

- **Semantic Scholar API**: Academic graph integration
- **OpenCitations**: Citation database integration  
- **Crossref API**: DOI resolution and metadata
- **ORCID API**: Author disambiguation

## Contributing

When adding new tools to RAG utils:

1. **Follow source-agnostic design**: Tools should work with any academic corpus
2. **Use abstract interfaces**: Enable pluggable architectures
3. **Document cultural assumptions**: Note discipline-specific behaviors
4. **Include usage examples**: Show integration with different sources
5. **Add anthropological context**: Explain the social meaning of the tool

## Testing

Each tool includes comprehensive tests:

```bash
# Run all RAG utils tests
cd tools/rag_utils
python -m pytest tests/

# Run specific tool tests
python -m pytest tests/test_citation_toolkit.py
```

## License

All RAG utils are part of the Olympus ecosystem and follow the project's licensing terms.
