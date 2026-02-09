# ArXiv Tools

ArXiv API client used by HADES for paper metadata fetching and abstract sync.

## Structure

```text
tools/arxiv/
├── arxiv_api_client.py   # ArXiv API interactions (search, metadata, download)
└── __init__.py
```

## Usage

```python
from core.tools.arxiv.arxiv_api_client import ArXivAPIClient

client = ArXivAPIClient()
metadata = client.get_metadata("2401.12345")
pdf_path = client.download_pdf("2401.12345", output_dir="/tmp")
```

## Note

The ArXiv manager, metadata config, and CLI search commands have been migrated
to a standalone [arxiv-manager](https://github.com/toddwbucy/arxiv-manager) tool.
See HADES Issue #83.
