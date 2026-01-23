# ArXiv Tools

Minimal ArXiv-specific processing tools for HADES-Lab.

## Structure

```text
tools/arxiv/
├── arxiv_manager.py      # ArXiv document manager using core workflows
├── arxiv_api_client.py   # ArXiv API interactions
├── arxiv_pipeline.py     # ACID pipeline for ArXiv papers
└── configs/              # ArXiv-specific configurations
```

## Usage

### Process ArXiv Papers

```bash
# Set environment
export ARANGO_PASSWORD="your-password"
export CUDA_VISIBLE_DEVICES=0,1

# Run ACID pipeline
python arxiv_pipeline.py \
    --config configs/acid_pipeline_phased.yaml \
    --count 100 \
    --arango-password "$ARANGO_PASSWORD"
```

### ArXiv Manager

The `arxiv_manager.py` provides high-level ArXiv document processing:

```python
from core.tools.arxiv.arxiv_manager import ArXivManager

manager = ArXivManager(config)
result = await manager.process_paper("2401.12345", "/path/to/pdf")
```

### ArXiv API Client

The `arxiv_api_client.py` handles ArXiv API interactions:

```python
from core.tools.arxiv.arxiv_api_client import ArXivAPIClient

client = ArXivAPIClient()
metadata = client.get_metadata("2401.12345")
pdf_path = client.download_pdf("2401.12345", output_dir="/tmp")
```

## Dependencies

All generic functionality comes from `core/`:
- Document processing: `core/workflows/`
- Database operations: `core/database/arango/`
- Embedders/Extractors: `core/framework/`

## Configuration

See `configs/acid_pipeline_phased.yaml` for pipeline configuration options.
