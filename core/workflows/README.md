# Workflows - Pipeline Orchestration

Base classes and utilities for orchestrating document processing pipelines with state persistence and checkpointing.

## Architecture

```
workflows/
├── workflow_base.py       # Abstract base class for workflows
├── workflow_pdf.py        # Simple PDF processing workflow
├── state/                 # State persistence
│   └── state_manager.py   # StateManager and CheckpointManager
├── storage/               # Storage abstractions
│   ├── storage_base.py    # Abstract storage interface
│   └── storage_local.py   # Local filesystem storage
└── __init__.py
```

## Components

### WorkflowBase

Abstract base class providing:
- Configuration management (`WorkflowConfig`)
- Atomic checkpoint save/load
- Result tracking (`WorkflowResult`)

```python
from core.workflows import WorkflowBase, WorkflowConfig, WorkflowResult

class CustomWorkflow(WorkflowBase):
    def validate_inputs(self, **kwargs) -> bool:
        return "input_path" in kwargs

    def execute(self, **kwargs) -> WorkflowResult:
        # Your processing logic here
        pass
```

### StateManager

Persistent state for long-running processes with atomic saves:

```python
from core.workflows.state import StateManager

state = StateManager(
    state_file="/tmp/pipeline_state.json",
    process_name="my_pipeline"
)

# Track progress
state.set_checkpoint("current_batch", 42)
state.increment_stat("processed", 10)
state.save()

# Resume after interruption
batch = state.get_checkpoint("current_batch", default=0)
```

### CheckpointManager

Tracks which items have been processed for resumable workflows:

```python
from core.workflows.state import CheckpointManager

checkpoint = CheckpointManager("/tmp/processed_items.json")

# Mark items as processed
checkpoint.mark_processed("doc_001")
checkpoint.mark_many_processed(["doc_002", "doc_003"])
checkpoint.save()

# Filter to unprocessed items
pending = checkpoint.get_unprocessed(all_document_ids)
```

### StorageBase

Abstract interface for workflow output storage:

```python
from core.workflows.storage import StorageBase

class MyStorage(StorageBase):
    def store(self, key: str, data: Any, metadata: dict = None) -> bool:
        # Store implementation
        pass

    def retrieve(self, key: str) -> Any:
        # Retrieve implementation
        pass

    # Also: exists(), delete(), list_keys()
```

### LocalStorage

Filesystem-based storage implementation:

```python
from core.workflows.storage import LocalStorage

storage = LocalStorage("/data/workflow_output")
storage.store("result_001", {"chunks": [...], "embeddings": [...]})
data = storage.retrieve("result_001")
```

## Configuration

```python
from core.workflows import WorkflowConfig

config = WorkflowConfig(
    name="pdf_processing",
    batch_size=32,
    num_workers=4,
    use_gpu=True,
    checkpoint_enabled=True,
    checkpoint_interval=100,
    timeout_seconds=300
)
```

## Usage Pattern

```python
from core.workflows import WorkflowBase, WorkflowConfig, WorkflowResult
from core.workflows.state import StateManager
from datetime import datetime

class MyWorkflow(WorkflowBase):
    def __init__(self, config: WorkflowConfig):
        super().__init__(config)
        self.state = StateManager(
            f"/tmp/{config.name}_state.json",
            config.name
        )

    def validate_inputs(self, input_dir: str = None, **kwargs) -> bool:
        return input_dir is not None

    def execute(self, input_dir: str, **kwargs) -> WorkflowResult:
        start_time = datetime.now()
        processed = 0
        failed = 0
        errors = []

        # Resume from checkpoint if available
        start_index = self.state.get_checkpoint("last_index", 0)

        for i, item in enumerate(items[start_index:], start=start_index):
            try:
                self.process_item(item)
                processed += 1

                # Periodic checkpoint
                if processed % self.config.checkpoint_interval == 0:
                    self.state.set_checkpoint("last_index", i)
                    self.state.save()
            except Exception as e:
                failed += 1
                errors.append(str(e))

        return WorkflowResult(
            workflow_name=self.name,
            success=failed == 0,
            items_processed=processed,
            items_failed=failed,
            start_time=start_time,
            end_time=datetime.now(),
            errors=errors
        )
```

## Related

- `core/processors/` - Document processing pipeline
- `core/extractors/` - Content extraction
- `core/embedders/` - Vector embedding generation
