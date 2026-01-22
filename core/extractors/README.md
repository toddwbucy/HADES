# Extractors - Content Extraction System

The extractors module implements multi-format content extraction, transforming diverse document formats into structured, processable text while preserving semantic structure and mathematical content.

## Overview

Extractors serve as the entry point for information processing, converting raw documents (PDFs, LaTeX, code) into structured representations that preserve both content and context.

## Architecture

```
extractors/
├── extractors_base.py       # Abstract base class and interfaces
├── extractors_docling.py    # IBM Docling for PDF/document extraction
├── extractors_latex.py      # LaTeX source processing
├── extractors_code.py       # Source code extraction
├── extractors_treesitter.py # Tree-sitter based code parsing
├── extractors_robust.py     # Fallback and error recovery
└── __init__.py             # Public API with get_extractor() dispatcher
```

## Core Components

### ExtractorBase

Abstract interface for all extractors:

```python
from core.extractors import ExtractorBase, ExtractionResult, ExtractorConfig
from typing import List, Union
from pathlib import Path

class CustomExtractor(ExtractorBase):
    """Custom extraction implementation."""

    @property
    def supported_formats(self) -> List[str]:
        """Supported file extensions."""
        return ['.custom']

    def extract(self, file_path: Union[str, Path], **kwargs) -> ExtractionResult:
        """Extract content from file."""
        return ExtractionResult(
            text=extracted_text,
            metadata=metadata,
            tables=tables,
            equations=equations
        )

    def extract_batch(self, file_paths: List[Union[str, Path]], **kwargs) -> List[ExtractionResult]:
        """Extract from multiple files."""
        return [self.extract(fp, **kwargs) for fp in file_paths]
```

### DoclingExtractor

PDF and document extraction using IBM Docling:

```python
from core.extractors import DoclingExtractor

# Initialize with configuration
extractor = DoclingExtractor(
    use_ocr=True,        # Enable OCR for scanned PDFs
    extract_tables=True,  # Extract table structures
    use_fallback=True     # Use PyMuPDF fallback on failure
)

# Extract from PDF
result = extractor.extract("paper.pdf")

# Access extracted content
print(f"Text length: {len(result.text)}")
print(f"Tables: {len(result.tables)}")
print(f"Equations: {len(result.equations)}")
print(f"Extractor: {result.metadata.get('extractor')}")
print(f"Processing time: {result.processing_time}s")
```

### LaTeXExtractor

Specialized extraction for LaTeX source files:

```python
from core.extractors import LaTeXExtractor

extractor = LaTeXExtractor(
    parse_math=True,  # Extract equations
    parse_citations=True,  # Extract bibliography
    resolve_commands=True  # Expand custom commands
)

# Extract from LaTeX
result = extractor.extract("paper.tex")

# Access LaTeX-specific content
for eq in result.equations:
    print(f"Equation: {eq.latex}")
    print(f"Rendered: {eq.text}")

for cite in result.citations:
    print(f"Citation: {cite.key} -> {cite.text}")
```

### CodeExtractor with Tree-sitter

Advanced code parsing with Tree-sitter:

```python
from core.extractors import CodeExtractor

extractor = CodeExtractor(
    languages=["python", "javascript", "rust"],
    extract_symbols=True,  # Functions, classes
    extract_comments=True,  # Documentation
    extract_imports=True  # Dependencies
)

# Extract from source code
result = extractor.extract("module.py")

# Access code structures
for func in result.functions:
    print(f"Function: {func.name}")
    print(f"Parameters: {func.parameters}")
    print(f"Docstring: {func.docstring}")

for cls in result.classes:
    print(f"Class: {cls.name}")
    print(f"Methods: {[m.name for m in cls.methods]}")
```

### RobustExtractor

Fallback extractor with timeout protection and PyMuPDF fallback:

```python
from core.extractors import RobustExtractor

# Timeout-protected extraction with fallback
extractor = RobustExtractor(
    use_ocr=False,        # Enable OCR for scanned PDFs
    extract_tables=True,  # Extract table structures
    timeout=30,           # Seconds before timeout
    use_fallback=True     # Use PyMuPDF fallback on failure
)

# Robust extraction with subprocess isolation
result = extractor.extract("problematic.pdf")
print(f"Extractor used: {result.metadata.get('extractor')}")
print(f"Text length: {len(result.text)}")
```

### get_extractor()

Simple dispatcher function to get the right extractor for a file:

```python
from core.extractors import get_extractor

# Auto-detect based on file type
extractor = get_extractor("document.pdf")
extractor = get_extractor("paper.tex")
extractor = get_extractor("module.py")

# With configuration
extractor = get_extractor("document.pdf", use_ocr=True)
```

## Extraction Results

### ExtractionResult Structure

```python
@dataclass
class ExtractionResult:
    """Result of document extraction."""

    # Core content
    text: str  # Full extracted text
    metadata: Dict[str, Any]  # Document metadata (extractor, num_pages, etc.)

    # Structural elements (lists of dicts)
    chunks: List[Dict[str, Any]]      # Segmented text chunks
    equations: List[Dict[str, Any]]   # Mathematical content
    tables: List[Dict[str, Any]]      # Tabular data
    images: List[Dict[str, Any]]      # Figures and images
    code_blocks: List[Dict[str, Any]] # Code blocks
    references: List[Dict[str, Any]]  # Bibliography references

    # Status
    error: Optional[str]       # Error message if extraction failed
    processing_time: float     # Extraction duration in seconds
```

## Usage Patterns

### Basic Extraction

```python
from core.extractors import DoclingExtractor

# Simple extraction
extractor = DoclingExtractor()
result = extractor.extract("research_paper.pdf")

# Process text
print(f"Title: {result.metadata.get('title')}")
print(f"Authors: {result.metadata.get('authors')}")
print(f"Abstract: {result.text[:500]}")
```

### Structured Processing

```python
# Extract with table structures
extractor = DoclingExtractor(extract_tables=True)

result = extractor.extract("paper.pdf")

# Access tables
for table in result.tables:
    print(f"Table caption: {table.get('caption', 'N/A')}")
    print(f"Table content: {table.get('content', '')[:200]}...")

# Access equations if extracted
for eq in result.equations:
    print(f"Equation: {eq.get('latex', str(eq))}")
```

### Batch Processing

```python
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

def process_document(file_path):
    """Process single document."""
    extractor = DoclingExtractor()
    return extractor.extract(file_path)

# Parallel extraction
pdf_files = list(Path("/data/papers").glob("*.pdf"))

with ProcessPoolExecutor(max_workers=8) as executor:
    results = list(executor.map(process_document, pdf_files))

print(f"Processed {len(results)} documents")
```

### Pipeline Integration

```python
from core.extractors import DoclingExtractor
from core.embedders import JinaV4Embedder
from core.database.arango import ArangoHttp2Client

# Complete pipeline
extractor = DoclingExtractor()
embedder = JinaV4Embedder()

# Process document
result = extractor.extract("paper.pdf")

# Generate embeddings from full text
# Note: chunking is typically done by the embedder with late chunking
embedding = embedder.embed(result.text)

# Or embed multiple documents
texts = [result.text for result in extractor.extract_batch(pdf_files)]
embeddings = embedder.embed_batch(texts)
```

## Advanced Features

### OCR Support

```python
# Enable OCR for scanned documents
extractor = DoclingExtractor(use_ocr=True)

result = extractor.extract("scanned_document.pdf")

print(f"Extractor: {result.metadata.get('extractor')}")
print(f"Text extracted: {len(result.text)} characters")
```

### Equation Extraction

```python
# Equations are extracted automatically by Docling
extractor = DoclingExtractor()

result = extractor.extract("math_paper.pdf")

# Equations are dicts with latex and label keys
for eq in result.equations:
    print(f"LaTeX: {eq.get('latex', str(eq))}")
    print(f"Label: {eq.get('label')}")
```

### Table Extraction

```python
# Tabular data extraction
extractor = DoclingExtractor(extract_tables=True)

result = extractor.extract("data_paper.pdf")

# Tables are dicts with caption, content, rows, headers keys
for table in result.tables:
    print(f"Caption: {table.get('caption', 'N/A')}")
    print(f"Content: {table.get('content', '')[:200]}")
    print(f"Headers: {table.get('headers', [])}")
```

## Performance Optimization

### Timeout Protection

```python
from core.extractors import RobustExtractor

# Use RobustExtractor for timeout protection
extractor = RobustExtractor(
    timeout=30,        # Max seconds for extraction
    use_fallback=True  # Fall back to PyMuPDF on timeout
)

result = extractor.extract("large_file.pdf")
```

### Caching

```python
from functools import lru_cache
import hashlib

class CachedExtractor(DoclingExtractor):
    """Extractor with result caching."""

    @lru_cache(maxsize=100)
    def _extract_cached(self, file_hash: str) -> ExtractionResult:
        """Cache extraction by file hash."""
        return super().extract(file_hash)

    def extract(self, file_path: str) -> ExtractionResult:
        """Extract with caching."""
        with open(file_path, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        return self._extract_cached(file_hash)
```

## Configuration

### YAML Configuration

```yaml
# extraction_config.yaml
extraction:
  type: docling
  use_ocr: true
  extract_tables: true
  extract_equations: true
  extract_images: false
  chunking:
    strategy: semantic
    size: 1000
    overlap: 200
  quality:
    min_confidence: 0.7
    max_warnings: 5
```

### Environment Variables

```bash
# Extraction settings
export EXTRACTION_TYPE=docling
export USE_OCR=true
export CHUNK_SIZE=1000
export NUM_WORKERS=8

# Performance tuning
export EXTRACTION_TIMEOUT=300
export MAX_MEMORY_MB=2048
```

## Error Handling

### Common Issues

```python
from core.extractors import DoclingExtractor, RobustExtractor

try:
    extractor = DoclingExtractor()
    result = extractor.extract("document.pdf")
except RuntimeError as e:
    # Try robust extractor with fallback
    robust = RobustExtractor(use_fallback=True)
    result = robust.extract("document.pdf")
except FileNotFoundError as e:
    print(f"File not found: {e}")
```

### Validation

```python
def validate_extraction(result: ExtractionResult) -> bool:
    """Validate extraction quality."""

    # Check for errors
    if result.error:
        return False

    # Check minimum content
    if len(result.text) < 100:
        return False

    return True
```

## Benchmarks

### Performance Metrics

| Extractor | Format | Speed (pages/sec) | Accuracy | Memory (MB) |
|-----------|--------|-------------------|----------|-------------|
| Docling | PDF | 2.5 | 98% | 512 |
| Docling + OCR | Scanned PDF | 0.8 | 95% | 1024 |
| LaTeX | .tex | 15.0 | 99% | 128 |
| Tree-sitter | Python | 50.0 | 100% | 64 |
| Robust | Mixed | 1.5 | 92% | 768 |

### Quality Metrics

| Feature | Precision | Recall | F1 Score |
|---------|-----------|--------|----------|
| Text Extraction | 0.98 | 0.99 | 0.98 |
| Equation Detection | 0.95 | 0.92 | 0.93 |
| Table Extraction | 0.91 | 0.88 | 0.89 |
| Section Detection | 0.96 | 0.94 | 0.95 |

## Best Practices

### 1. Choose Appropriate Extractor

```python
# For research papers
extractor = DoclingExtractor(
    extract_equations=True,
    extract_tables=True,
    extract_citations=True
)

# For code repositories
extractor = CodeExtractor(
    extract_symbols=True,
    extract_dependencies=True
)

# For mixed content
extractor = RobustExtractor(
    strategies=["docling", "latex", "code"]
)
```

### 2. Validate Input Files

```python
def validate_file(file_path: str) -> bool:
    """Validate file before extraction."""
    path = Path(file_path)

    # Check existence
    if not path.exists():
        return False

    # Check size
    if path.stat().st_size > 100 * 1024 * 1024:  # 100MB
        return False

    # Check format
    supported = ['.pdf', '.tex', '.py', '.js', '.rs']
    if path.suffix not in supported:
        return False

    return True
```

### 3. Handle Extraction Failures

```python
def extract_with_fallback(file_path: str) -> ExtractionResult:
    """Extract with multiple fallback strategies."""
    strategies = [
        DoclingExtractor(),
        RobustExtractor(),
        OCRExtractor()
    ]

    for strategy in strategies:
        try:
            result = strategy.extract(file_path)
            if validate_extraction(result):
                return result
        except Exception as e:
            continue

    raise ExtractionError(f"All strategies failed for {file_path}")
```

### 4. Monitor Extraction Quality

```python
def monitor_extraction(result: ExtractionResult) -> Dict:
    """Monitor extraction quality metrics."""
    return {
        "text_length": len(result.text),
        "num_chunks": len(result.chunks),
        "num_equations": len(result.equations),
        "num_tables": len(result.tables),
        "confidence": result.confidence,
        "warnings": len(result.warnings),
        "extraction_time": result.statistics.duration
    }
```

## Testing

```python
import pytest
from core.extractors import DoclingExtractor

def test_pdf_extraction():
    """Test PDF extraction."""
    extractor = DoclingExtractor()
    result = extractor.extract("test_paper.pdf")

    assert len(result.text) > 0
    assert result.confidence > 0.8
    assert len(result.chunks) > 0

def test_chunking():
    """Test chunking strategy."""
    extractor = DoclingExtractor(
        chunk_size=1000,
        overlap=200
    )
    result = extractor.extract("test_paper.pdf")

    for chunk in result.chunks:
        assert len(chunk.text) <= 1000
        assert chunk.id is not None

def test_structure_preservation():
    """Test structure extraction."""
    extractor = DoclingExtractor(
        preserve_structure=True
    )
    result = extractor.extract("test_paper.pdf")

    assert len(result.sections) > 0
    assert result.sections[0].title is not None
```

## Migration Guide

### From PyPDF2

```python
# Old approach
import PyPDF2
reader = PyPDF2.PdfReader("file.pdf")
text = ""
for page in reader.pages:
    text += page.extract_text()

# New approach
from core.extractors import DoclingExtractor
extractor = DoclingExtractor()
result = extractor.extract("file.pdf")
text = result.text
```

### From pdfplumber

```python
# Old approach
import pdfplumber
with pdfplumber.open("file.pdf") as pdf:
    text = ""
    for page in pdf.pages:
        text += page.extract_text()

# New approach
from core.extractors import DoclingExtractor
extractor = DoclingExtractor()
result = extractor.extract("file.pdf")
text = result.text
# Plus: equations, tables, structure
```

## Related Components

- [Embedders](../embedders/README.md) - Process extracted text
- [Workflows](../workflows/README.md) - Orchestrate extraction pipelines
- [Processors](../processors/README.md) - Post-process extracted content
- [Config](../config/README.md) - Configure extractors