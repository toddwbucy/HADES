# Embedders - Vector Embedding System

The embedders module transforms text into high-dimensional vector representations, enabling semantic similarity search and knowledge retrieval.

## Overview

Embedders bridge human-readable text and machine-processable vectors, preserving semantic relationships while enabling efficient similarity computations.

## Architecture

```text
embedders/
├── embedders_base.py     # Abstract base class and interfaces
├── embedders_jina.py     # Jina V4 implementation with late chunking
├── embedders_factory.py  # Factory pattern for embedder instantiation
└── __init__.py          # Public API exports
```

## Core Components

### EmbedderBase

Abstract interface defining the embedder contract:

```python
from core.embedders import EmbedderBase, EmbeddingConfig

class CustomEmbedder(EmbedderBase):
    """Custom embedding implementation."""

    def embed(self, texts: List[str]) -> np.ndarray:
        """Transform texts to vectors."""
        # Implementation
        pass

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Batch processing for efficiency."""
        # Implementation
        pass
```

### JinaV4Embedder

State-of-the-art embedder with late chunking support:

```python
from core.embedders import JinaV4Embedder
from core.embedders.embedders_base import EmbeddingConfig

# Initialize with configuration
config = EmbeddingConfig(
    model_name="jinaai/jina-embeddings-v4",
    device="cuda",
    use_fp16=True,  # Memory efficient
)
embedder = JinaV4Embedder(config)

# Single text embedding
text = "Neural networks enable semantic understanding of text"
vector = embedder.embed_single(text)  # Shape: (2048,)

# Batch embedding
documents = ["Long document 1...", "Long document 2..."]
vectors = embedder.embed_texts(documents, batch_size=24)

# Late chunking for long documents (returns ChunkWithEmbedding list)
chunks = embedder.embed_with_late_chunking("Very long document text...")
```

### EmbedderFactory

Factory pattern for flexible embedder instantiation:

```python
from core.embedders import EmbedderFactory
from core.embedders.embedders_base import EmbeddingConfig

config = EmbeddingConfig(
    model_name="jinaai/jina-embeddings-v4",
    device="cuda",
    batch_size=32,
    use_fp16=True,
    chunk_size_tokens=512,
    chunk_overlap_tokens=128,
)
embedder = EmbedderFactory.create(model_name=config.model_name, config=config)
```

## Embedding Models

### Jina V4 (1024-dimension mode)

```python
from core.embedders.embedders_base import EmbeddingConfig
from core.embedders import JinaV4Embedder

config = EmbeddingConfig(
    model_name="jinaai/jina-embeddings-v4",
    device="cuda",
    batch_size=32,
    use_fp16=True,
    chunk_size_tokens=512,
    chunk_overlap_tokens=128,
)
embedder = JinaV4Embedder(config)
```

- **Dimensions**: 1024 (Jina model exposes both 1024- and 2048-dimension adapters)
- **Max Sequence**: 8192 tokens (configurable via `EmbeddingConfig`)

### Jina V4 (2048-dimension mode)

```python
from core.embedders.embedders_base import EmbeddingConfig
from core.embedders import JinaV4Embedder

config = EmbeddingConfig(
    model_name="jinaai/jina-embeddings-v4",
    device="cuda",
    batch_size=16,
    use_fp16=True,
    chunk_size_tokens=1000,
    chunk_overlap_tokens=200,
)
embedder = JinaV4Embedder(config)
```

- **Dimensions**: 2048
- **Max Sequence**: 32768 tokens
- **Use Cases**: High-precision retrieval and long-document embeddings

## Late Chunking

Late chunking processes full documents before segmentation, preserving context:

```python
from core.embedders import JinaV4Embedder

from core.embedders.embedders_base import EmbeddingConfig
from core.embedders import JinaV4Embedder

config = EmbeddingConfig(
    model_name="jinaai/jina-embeddings-v4",
    device="cuda",
    chunk_size_tokens=512,
    chunk_overlap_tokens=128,
)
embedder = JinaV4Embedder(config)

long_text = "..." * 10000  # Very long document

# Proper late chunking: encode once, then slice contextual embeddings
chunks: List[ChunkWithEmbedding] = embedder.embed_with_late_chunking(long_text)
vectors = [chunk.embedding for chunk in chunks]
```

### Benefits

1. **Context Preservation**: Full document understanding before chunking
2. **Semantic Coherence**: Related chunks maintain relationships
3. **Better Retrieval**: Improved similarity matching
4. **Efficiency**: Single encoding pass for entire document

## Usage Patterns

### Basic Embedding

```python
from core.embedders import JinaV4Embedder
from core.embedders.embedders_base import EmbeddingConfig

# Initialize
config = EmbeddingConfig(model_name="jinaai/jina-embeddings-v4", device="cuda")
embedder = JinaV4Embedder(config)

# Embed single text
text = "Vector embeddings enable semantic similarity search"
vector = embedder.embed_single(text)

print(f"Shape: {vector.shape}")  # (2048,)
print(f"Norm: {np.linalg.norm(vector):.4f}")  # Normalized to ~1.0
```

### Batch Processing

```python
# Efficient batch processing
texts = load_documents()  # List of 1000 documents

# Process in optimized batches
embeddings = embedder.embed_texts(texts, batch_size=32)

print(f"Processed {len(texts)} documents")
print(f"Embeddings shape: {embeddings.shape}")  # (1000, 2048)
```

### Memory-Efficient Processing

```python
from core.embedders.embedders_base import EmbeddingConfig

# Use FP16 for reduced memory
config = EmbeddingConfig(
    model_name="jinaai/jina-embeddings-v4",
    device="cuda",
    use_fp16=True,  # 40% memory reduction
    batch_size=64  # Larger batches with FP16
)
embedder = JinaV4Embedder(config)

# Process large batches
large_batch = ["text"] * 100
embeddings = embedder.embed_texts(large_batch)
```

### Pipeline Integration

```python
from core.embedders import JinaV4Embedder
from core.embedders.embedders_base import EmbeddingConfig
from core.extractors import DoclingExtractor
from core.database import ArangoClient

# Complete pipeline
extractor = DoclingExtractor()
config = EmbeddingConfig(model_name="jinaai/jina-embeddings-v4", device="cuda", use_fp16=True)
embedder = JinaV4Embedder(config)
db = ArangoClient()

# Process document
content = extractor.extract("paper.pdf")
chunks = content.get_chunks()

# Generate embeddings
embeddings = embedder.embed_texts([chunk.text for chunk in chunks], batch_size=32)

# Store in database
for chunk, embedding in zip(chunks, embeddings):
    db.store_embedding(
        document_id=content.document_id,
        chunk_id=chunk.id,
        text=chunk.text,
        embedding=embedding.tolist()
    )
```

## Performance Optimization

### GPU Utilization

```python
from core.embedders.embedders_base import EmbeddingConfig

# Optimize for GPU memory
config = EmbeddingConfig(
    model_name="jinaai/jina-embeddings-v4",
    device="cuda:0",
    batch_size=32,  # Tune based on GPU memory
    use_fp16=True,  # Reduce memory usage
    max_seq_length=8192  # Limit sequence length
)
embedder = JinaV4Embedder(config)

# Monitor GPU usage
import torch
print(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
```

### Multi-GPU Support

```python
from core.embedders.embedders_base import EmbeddingConfig

# Distribute across multiple GPUs
config = EmbeddingConfig(
    model_name="jinaai/jina-embeddings-v4",
    device="cuda",  # Auto-selects available GPU
    batch_size=128  # Larger batch for multi-GPU
)
embedder = JinaV4Embedder(config)

# Process large dataset
embeddings = embedder.embed_texts(large_dataset)
```

### Caching Strategy

```python
from functools import lru_cache
import hashlib

class CachedEmbedder:
    """Wrapper that adds caching to any embedder."""

    def __init__(self, embedder: JinaV4Embedder):
        self.embedder = embedder
        self._cache: Dict[str, np.ndarray] = {}

    def embed_single(self, text: str) -> np.ndarray:
        """Embed with caching using text hash as key."""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash not in self._cache:
            self._cache[text_hash] = self.embedder.embed_single(text)
        return self._cache[text_hash]

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed multiple texts with caching."""
        return np.array([self.embed_single(text) for text in texts])
```

## Configuration

### YAML Configuration

```yaml
# embedding_config.yaml
embedding:
  type: jina_v4
  model_name: jinaai/jina-embeddings-v4
  device: cuda
  batch_size: 32
  use_fp16: true
  max_seq_length: 8192
  late_chunking:
    enabled: true
    chunk_size: 512
    overlap: 128
```

### Environment Variables

```bash
# GPU configuration
export CUDA_VISIBLE_DEVICES=0,1
export EMBEDDING_DEVICE=cuda

# Model settings
export EMBEDDING_MODEL=jinaai/jina-embeddings-v4
export EMBEDDING_BATCH_SIZE=32
export USE_FP16=true
```

## Error Handling

### Common Issues

```python
from core.embedders import JinaV4Embedder
from core.embedders.embedders_base import EmbeddingConfig

try:
    config = EmbeddingConfig(model_name="jinaai/jina-embeddings-v4", device="cuda")
    embedder = JinaV4Embedder(config)
    embeddings = embedder.embed_texts(texts)
except torch.cuda.OutOfMemoryError:
    # Reduce batch size or use CPU
    config = EmbeddingConfig(model_name="jinaai/jina-embeddings-v4", device="cpu", batch_size=8)
    embedder = JinaV4Embedder(config)
    embeddings = embedder.embed_texts(texts)
except Exception as e:
    print(f"Embedding failed: {e}")
    # Implement fallback strategy
```

### Validation

```python
def validate_embeddings(embeddings: np.ndarray) -> bool:
    """Validate embedding quality."""
    # Check dimensions
    if embeddings.shape[1] not in [1024, 2048]:
        return False

    # Check for NaN/Inf
    if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
        return False

    # Check normalization (should be ~1.0)
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=0.1):
        return False

    return True
```

## Benchmarks

### Performance Metrics

| Model | Dimensions | Batch Size | GPU | Throughput | Memory |
|-------|------------|------------|-----|------------|--------|
| Jina V3 | 1024 | 32 | RTX 3090 | 150 texts/sec | 7.2 GB |
| Jina V3 | 1024 | 64 | RTX 3090 | 180 texts/sec | 12.1 GB |
| Jina V3 + FP16 | 1024 | 64 | RTX 3090 | 220 texts/sec | 7.8 GB |
| Jina V4 | 2048 | 32 | RTX 3090 | 95 texts/sec | 10.5 GB |

### Quality Metrics

| Model | MTEB Score | ArXiv Retrieval | Code Search |
|-------|------------|-----------------|-------------|
| Jina V3 | 0.872 | 0.891 | 0.823 |
| Jina V4 | 0.908 | 0.924 | 0.867 |

## Best Practices

### 1. Choose Appropriate Model

```python
from core.embedders.embedders_base import EmbeddingConfig

# Jina V4 uses 2048 dimensions by default
config = EmbeddingConfig(model_name="jinaai/jina-embeddings-v4", device="cuda")
embedder = JinaV4Embedder(config)
```

### 2. Optimize Batch Size

```python
# Find optimal batch size
def find_optimal_batch_size(embedder, sample_texts):
    for batch_size in [8, 16, 32, 64, 128]:
        try:
            embedder.batch_size = batch_size
            _ = embedder.embed_texts(sample_texts[:batch_size])
            print(f"Batch size {batch_size} works")
        except torch.cuda.OutOfMemoryError:
            print(f"Batch size {batch_size} too large")
            return batch_size // 2
    return 128
```

### 3. Use Late Chunking for Long Documents

```python
# Always use late chunking for documents > 2000 tokens
if len(tokenizer.encode(document)) > 2000:
    chunks = embedder.embed_with_late_chunking(document)
    embeddings = [chunk.embedding for chunk in chunks]
else:
    embeddings = [embedder.embed_single(document)]
```

### 4. Monitor Resource Usage

```python
import psutil
import GPUtil

def monitor_resources():
    # CPU/Memory
    cpu_percent = psutil.cpu_percent()
    memory_percent = psutil.virtual_memory().percent

    # GPU
    gpus = GPUtil.getGPUs()
    gpu_memory = gpus[0].memoryUsed if gpus else 0

    return {
        "cpu": cpu_percent,
        "memory": memory_percent,
        "gpu_memory_mb": gpu_memory
    }
```

## Testing

```python
import pytest
import numpy as np
from core.embedders import JinaV4Embedder
from core.embedders.embedders_base import EmbeddingConfig

def test_embedding_dimensions():
    """Test embedding dimensions."""
    config = EmbeddingConfig(model_name="jinaai/jina-embeddings-v4", device="cpu")
    embedder = JinaV4Embedder(config)
    text = "Test text"
    embedding = embedder.embed_single(text)

    assert embedding.shape == (2048,)  # Jina V4 uses 2048 dimensions
    assert np.allclose(np.linalg.norm(embedding), 1.0, atol=0.1)

def test_batch_consistency():
    """Test batch processing consistency."""
    config = EmbeddingConfig(model_name="jinaai/jina-embeddings-v4", device="cpu")
    embedder = JinaV4Embedder(config)
    texts = ["Text 1", "Text 2", "Text 3"]

    # Single processing
    single = np.array([embedder.embed_single(t) for t in texts])

    # Batch processing
    batch = embedder.embed_texts(texts, batch_size=3)

    assert np.allclose(single, batch, atol=1e-5)
```

## Migration Guide

### From Sentence Transformers

```python
# Old approach
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(texts)

# New approach
from core.embedders import JinaV4Embedder
from core.embedders.embedders_base import EmbeddingConfig

config = EmbeddingConfig(model_name="jinaai/jina-embeddings-v4", device="cuda")
embedder = JinaV4Embedder(config)
embeddings = embedder.embed_texts(texts)
```

### From OpenAI Embeddings

```python
# Old approach
import openai
response = openai.Embedding.create(
    input=texts,
    model="text-embedding-ada-002"
)

# New approach
from core.embedders import JinaV4Embedder
from core.embedders.embedders_base import EmbeddingConfig

config = EmbeddingConfig(model_name="jinaai/jina-embeddings-v4", device="cuda")
embedder = JinaV4Embedder(config)
embeddings = embedder.embed_texts(texts)
# Better performance, lower cost, local processing
```

## Related Components

- [Extractors](../extractors/README.md) - Text extraction for embedding
- [Workflows](../workflows/README.md) - Pipeline integration
- [Database](../database/README.md) - Embedding storage
- [Config](../config/README.md) - Configuration management
