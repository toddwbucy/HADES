# Academic Citation Toolkit Documentation

## Overview

The Academic Citation Toolkit is a **source-agnostic** system for extracting citations and bibliographies from academic papers. It works with any academic corpus - ArXiv, SSRN, PubMed, Harvard Law Library, or any collection of academic papers.

## Key Classes and Interfaces

### Core Data Models

#### `BibliographyEntry`
Represents a single bibliography entry from any academic paper.

```python
@dataclass
class BibliographyEntry:
    entry_number: Optional[str]      # [1], [2], etc.
    raw_text: str                    # Full citation text
    authors: List[str]               # Extracted author names
    title: Optional[str]             # Paper title
    venue: Optional[str]             # Journal/conference
    year: Optional[int]              # Publication year
    arxiv_id: Optional[str]          # ArXiv identifier
    doi: Optional[str]               # DOI identifier
    pmid: Optional[str]              # PubMed ID
    ssrn_id: Optional[str]           # SSRN ID
    url: Optional[str]               # Web URL
    source_paper_id: str             # ID of citing paper
    confidence: float                # Parsing confidence (0-1)
```

#### `InTextCitation`  
Represents an in-text citation that points to a bibliography entry.

```python
@dataclass
class InTextCitation:
    raw_text: str                    # [1], (Smith, 2020), etc.
    citation_type: str               # 'numeric', 'author_year', 'hybrid'
    start_pos: int                   # Position in text
    end_pos: int                     # End position
    context: str                     # Surrounding text
    bibliography_ref: Optional[str]  # Points to bibliography entry
    source_paper_id: str             # ID of citing paper
    confidence: float                # Extraction confidence
```

### Abstract Interfaces

#### `DocumentProvider`
Abstract interface for providing document text from any source.

```python
class DocumentProvider(ABC):
    @abstractmethod
    def get_document_text(self, document_id: str) -> Optional[str]:
        """Get full text of a document by ID."""
        pass
    
    @abstractmethod  
    def get_document_chunks(self, document_id: str) -> List[str]:
        """Get text chunks of a document by ID."""
        pass
```

**Implementations**:
- `ArangoDocumentProvider`: For ArangoDB (our ArXiv setup)
- `FileSystemDocumentProvider`: For local files

#### `CitationStorage`
Abstract interface for storing citation data in any backend.

```python
class CitationStorage(ABC):
    @abstractmethod
    def store_bibliography_entries(self, entries: List[BibliographyEntry]) -> bool:
        """Store bibliography entries."""
        pass
    
    @abstractmethod
    def store_citations(self, citations: List[InTextCitation]) -> bool:
        """Store in-text citations."""
        pass
```

**Implementations**:
- `ArangoCitationStorage`: For ArangoDB storage  
- `JSONCitationStorage`: For JSON file storage

### Main Processing Classes

#### `UniversalBibliographyExtractor`
Extracts bibliography sections and parses entries from any academic paper.

```python
class UniversalBibliographyExtractor:
    def __init__(self, document_provider: DocumentProvider):
        self.document_provider = document_provider
    
    def extract_bibliography_section(self, paper_id: str) -> Optional[str]:
        """Extract bibliography/references section text."""
        
    def parse_bibliography_entries(self, bibliography_text: str, paper_id: str) -> List[BibliographyEntry]:
        """Parse individual bibliography entries."""
        
    def extract_paper_bibliography(self, paper_id: str) -> List[BibliographyEntry]:
        """Main entry point: extract complete bibliography for a paper."""
```

**Bibliography Extraction Strategies**:
1. **Explicit Headers**: Look for "References", "Bibliography", "Works Cited"
2. **Numbered Patterns**: Find `[1], [2], [3]...` reference lists
3. **Author-Year Patterns**: Detect `Author (Year)` format lists
4. **Mixed Format Detection**: Handle various academic disciplines

**Entry Parsing Features**:
- **Identifier Extraction**: ArXiv IDs, DOIs, PubMed IDs, SSRN IDs
- **Metadata Extraction**: Authors, titles, venues, years
- **Confidence Scoring**: Based on extracted information quality
- **Format Flexibility**: Works with different citation styles

#### `UniversalCitationExtractor`
Extracts in-text citations and maps them to bibliography entries.

```python
class UniversalCitationExtractor:
    def __init__(self, document_provider: DocumentProvider):
        self.document_provider = document_provider
    
    def extract_citations(self, paper_id: str, bibliography_entries: List[BibliographyEntry]) -> List[InTextCitation]:
        """Extract in-text citations and map to bibliography."""
```

## Usage Examples

### Quick Start with Factory Functions

#### ArXiv Papers (ArangoDB)
```python
from rag_utils.academic_citation_toolkit import create_arxiv_citation_toolkit
from arango import ArangoClient

client = ArangoClient(hosts='http://192.168.1.69:8529')
extractor, storage = create_arxiv_citation_toolkit(client)

# Extract bibliography from a paper
entries = extractor.extract_paper_bibliography("1301_3781")
print(f"Found {len(entries)} bibliography entries")

# Store results
storage.store_bibliography_entries(entries)
```

#### Local Files (JSON Output)
```python
from rag_utils.academic_citation_toolkit import create_filesystem_citation_toolkit

extractor, storage = create_filesystem_citation_toolkit(
    file_path="/path/to/papers",
    output_path="/path/to/results"
)

entries = extractor.extract_paper_bibliography("paper_001")
storage.store_bibliography_entries(entries)
```

### Advanced Usage with Custom Providers

#### Custom Document Provider
```python
from rag_utils.academic_citation_toolkit import (
    UniversalBibliographyExtractor,
    DocumentProvider,
    BibliographyEntry
)

class SSRNProvider(DocumentProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def get_document_text(self, document_id: str) -> Optional[str]:
        # Fetch from SSRN API
        response = requests.get(f"https://ssrn.com/api/papers/{document_id}", 
                              headers={"Authorization": f"Bearer {self.api_key}"})
        return response.json().get("full_text") if response.ok else None
    
    def get_document_chunks(self, document_id: str) -> List[str]:
        text = self.get_document_text(document_id)
        return text.split('\n\n') if text else []

# Use the custom provider
provider = SSRNProvider(api_key="your_key")
extractor = UniversalBibliographyExtractor(provider)
```

#### Custom Storage Backend
```python
from rag_utils.academic_citation_toolkit import CitationStorage
import sqlite3

class SQLiteCitationStorage(CitationStorage):
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS bibliography_entries (
            id INTEGER PRIMARY KEY,
            source_paper_id TEXT,
            entry_number TEXT,
            raw_text TEXT,
            title TEXT,
            authors TEXT,
            year INTEGER,
            confidence REAL
        )
        """)
        conn.commit()
        conn.close()
    
    def store_bibliography_entries(self, entries: List[BibliographyEntry]) -> bool:
        try:
            conn = sqlite3.connect(self.db_path)
            for entry in entries:
                conn.execute("""
                INSERT INTO bibliography_entries 
                (source_paper_id, entry_number, raw_text, title, authors, year, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (entry.source_paper_id, entry.entry_number, entry.raw_text,
                      entry.title, ','.join(entry.authors), entry.year, entry.confidence))
            conn.commit()
            return True
        except Exception as e:
            print(f"Error storing entries: {e}")
            return False
        finally:
            conn.close()
```

### Batch Processing Example
```python
def process_paper_batch(paper_ids: List[str], extractor, storage):
    """Process multiple papers efficiently."""
    all_entries = []
    
    for paper_id in paper_ids:
        print(f"Processing paper: {paper_id}")
        entries = extractor.extract_paper_bibliography(paper_id)
        all_entries.extend(entries)
        print(f"  Found {len(entries)} bibliography entries")
    
    # Batch store all entries
    if all_entries:
        success = storage.store_bibliography_entries(all_entries)
        print(f"Stored {len(all_entries)} total entries: {'✅' if success else '❌'}")
    
    return all_entries

# Usage
paper_ids = ["1301_3781", "1405_4053", "1803_09473"]  
entries = process_paper_batch(paper_ids, extractor, storage)
```

## Configuration and Customization

### Bibliography Extraction Tuning

The extractor uses multiple strategies with configurable parameters:

```python
class UniversalBibliographyExtractor:
    def extract_bibliography_section(self, paper_id: str) -> Optional[str]:
        # Strategy 1: Explicit headers (customizable patterns)
        references_patterns = [
            r'\b(References|Bibliography|REFERENCES|BIBLIOGRAPHY|Works Cited|Literature Cited)\b.*?(?=\n\n[A-Z][a-z]+|\Z)',
            r'## References(.*?)(?=\n##|\Z)',  # Markdown style
            r'# References(.*?)(?=\n#|\Z)'     # Markdown style
        ]
        
        # Strategy 2: Numbered references (configurable lookback)
        numbered_refs = re.search(
            r'(\[\d+\].*?)(?=\n\n[A-Z][a-z]+|\Z)', 
            full_text[-8000:],  # Configurable: last N chars
            re.DOTALL
        )
        
        # Strategy 3: Author-year format (configurable minimum entries)
        author_year_refs = re.search(
            r'((?:[A-Z][a-z]+(?:\s+et\s+al\.?)?,\s+\d{4}.*?\n){5,})',  # Min 5 entries
            full_text[-5000:],  # Configurable lookback
            re.DOTALL
        )
```

### Entry Parsing Customization

```python
def _parse_single_entry(self, entry_text: str, paper_id: str, entry_number: str = None) -> Optional[BibliographyEntry]:
    # Configurable confidence scoring
    confidence = 0.3  # Base confidence
    if arxiv_id or doi or pmid or ssrn_id:
        confidence += 0.4  # Strong identifiers
    if title and len(title) > 10:
        confidence += 0.2  # Good title extraction
    if authors:
        confidence += 0.2  # Author information
    if year:
        confidence += 0.1  # Publication year
    if venue:
        confidence += 0.1  # Venue information
    
    confidence = min(1.0, confidence)  # Cap at 100%
```

## Error Handling and Robustness

### Graceful Degradation
```python
def extract_paper_bibliography(self, paper_id: str) -> List[BibliographyEntry]:
    """Extract complete bibliography with error handling."""
    try:
        bibliography_text = self.extract_bibliography_section(paper_id)
        if not bibliography_text:
            logger.warning(f"No bibliography found for {paper_id}")
            return []
        
        entries = self.parse_bibliography_entries(bibliography_text, paper_id)
        logger.info(f"Successfully extracted {len(entries)} entries for {paper_id}")
        return entries
        
    except Exception as e:
        logger.error(f"Error processing bibliography for {paper_id}: {e}")
        return []  # Return empty list instead of crashing
```

### Confidence Scoring
Every extracted entry includes a confidence score (0.0 to 1.0):

- **0.8-1.0**: High confidence - has strong identifiers (DOI, ArXiv ID) and good metadata
- **0.6-0.8**: Medium confidence - has some identifiers and metadata
- **0.3-0.6**: Low confidence - minimal metadata, may need manual review
- **0.0-0.3**: Very low confidence - likely parsing errors

### Logging and Monitoring
```python
import logging

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Example log output:
# 2025-09-04 01:30:00 - academic_citation_toolkit - INFO - Found references section for 1301_3781
# 2025-09-04 01:30:01 - academic_citation_toolkit - INFO - Found 43 numbered bibliography entries  
# 2025-09-04 01:30:02 - academic_citation_toolkit - INFO - Successfully extracted 42 entries for 1301_3781
```

## Performance Optimization

### Memory Usage
- **Streaming processing**: Process papers individually, don't load entire corpus
- **Configurable chunk sizes**: Balance memory vs processing efficiency  
- **Lazy loading**: Only load document text when needed

### Processing Speed
- **Regex optimization**: Pre-compile frequently used patterns
- **Batch operations**: Group database operations for efficiency
- **Parallel processing**: Easy to distribute across workers

```python
# Example parallel processing
from concurrent.futures import ThreadPoolExecutor

def process_papers_parallel(paper_ids: List[str], extractor, storage, max_workers=4):
    def process_single(paper_id):
        return extractor.extract_paper_bibliography(paper_id)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(process_single, paper_ids)
    
    all_entries = [entry for result in results for entry in result]
    storage.store_bibliography_entries(all_entries)
    return all_entries
```

## Integration Patterns

### With HERMES Data Pipeline
```python
# Integration example for HERMES
from hermes.pipeline import Pipeline
from rag_utils.academic_citation_toolkit import create_arxiv_citation_toolkit

class CitationEnrichmentStage:
    def __init__(self, arango_client):
        self.extractor, self.storage = create_arxiv_citation_toolkit(arango_client)
    
    def process_document(self, document_id: str) -> Dict:
        entries = self.extractor.extract_paper_bibliography(document_id)
        self.storage.store_bibliography_entries(entries)
        
        return {
            'document_id': document_id,
            'bibliography_count': len(entries),
            'high_confidence_entries': len([e for e in entries if e.confidence >= 0.8])
        }

# Add to HERMES pipeline
pipeline = Pipeline()
pipeline.add_stage('citation_enrichment', CitationEnrichmentStage(arango_client))
```

### With HADES Dimensional Analysis
```python
# Integration example for HADES dimensional analysis
def calculate_conveyance_from_citations(bibliography_entries: List[BibliographyEntry]) -> float:
    """
    Calculate CONVEYANCE dimension based on citation patterns.
    
    Papers with more implementation-focused citations have higher conveyance.
    """
    implementation_keywords = ['algorithm', 'implementation', 'code', 'software', 'system']
    
    implementation_score = 0
    for entry in bibliography_entries:
        if entry.title:
            keyword_count = sum(1 for keyword in implementation_keywords 
                               if keyword.lower() in entry.title.lower())
            implementation_score += keyword_count
    
    # Normalize by number of entries
    if bibliography_entries:
        return min(1.0, implementation_score / len(bibliography_entries))
    return 0.0
```

## Testing and Validation

### Unit Tests
```python
import pytest
from rag_utils.academic_citation_toolkit import UniversalBibliographyExtractor

def test_bibliography_parsing():
    """Test bibliography entry parsing."""
    sample_text = """
    [1] Y. Bengio, R. Ducharme, P. Vincent. A neural probabilistic language model. 
    Journal of Machine Learning Research, 3:1137-1155, 2003.
    
    [2] T. Mikolov, K. Chen, G. Corrado, and J. Dean. Efficient Estimation of Word 
    Representations in Vector Space. arXiv:1301.3781, 2013.
    """
    
    # Mock document provider
    class MockProvider:
        def get_document_text(self, doc_id): return sample_text
        def get_document_chunks(self, doc_id): return [sample_text]
    
    extractor = UniversalBibliographyExtractor(MockProvider())
    entries = extractor.parse_bibliography_entries(sample_text, "test_paper")
    
    assert len(entries) == 2
    assert entries[0].entry_number == "1"
    assert "neural probabilistic language model" in entries[0].raw_text
    assert entries[1].arxiv_id == "1301.3781"
```

### Integration Tests
```python
def test_full_extraction_workflow():
    """Test complete extraction workflow with real data."""
    from arango import ArangoClient
    
    client = ArangoClient(hosts='http://192.168.1.69:8529')
    extractor, storage = create_arxiv_citation_toolkit(client)
    
    # Test with known paper
    entries = extractor.extract_paper_bibliography("1301_3781")
    assert len(entries) > 0
    assert all(entry.confidence > 0 for entry in entries)
    
    # Test storage
    success = storage.store_bibliography_entries(entries)
    assert success
```

This toolkit provides a robust, flexible foundation for building citation networks from any academic corpus, with comprehensive error handling, performance optimization, and integration capabilities.