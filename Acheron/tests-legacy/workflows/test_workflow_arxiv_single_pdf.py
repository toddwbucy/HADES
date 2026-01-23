import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from core.embedders.embedders_jina import ChunkWithEmbedding
from core.processors.document_processor import ExtractionResult, ProcessingResult
from core.workflows.workflow_arxiv_single_pdf import ArxivSinglePDFWorkflow
from core.workflows.workflow_base import WorkflowConfig


@dataclass
class StubMetadata:
    arxiv_id: str
    title: str
    abstract: str
    authors: list[str]
    categories: list[str]
    primary_category: str
    published: datetime
    updated: datetime
    has_pdf: bool = True
    has_latex: bool = False


@dataclass
class StubDownloadResult:
    success: bool
    arxiv_id: str
    pdf_path: Path | None
    latex_path: Path | None
    metadata: StubMetadata | None
    error_message: str | None = None
    file_size_bytes: int = 0


class DummyAPIClient:
    def __init__(self, tmp_dir: Path) -> None:
        self.tmp_dir = tmp_dir
        self.last_validated: str | None = None

    def validate_arxiv_id(self, arxiv_id: str) -> bool:
        self.last_validated = arxiv_id
        return True

    def download_paper(self, arxiv_id: str, pdf_dir: Path, latex_dir: Path | None, force: bool) -> StubDownloadResult:
        # Silence linter warnings for unused parameters (required by interface)
        _ = (latex_dir, force)
        pdf_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = pdf_dir / f"{arxiv_id}.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")

        metadata = StubMetadata(
            arxiv_id=arxiv_id,
            title="Test Document",
            abstract="Short abstract",
            authors=["Author One"],
            categories=["cs.AI"],
            primary_category="cs.AI",
            published=datetime.now(UTC),
            updated=datetime.now(UTC),
            has_pdf=True,
            has_latex=False,
        )

        return StubDownloadResult(
            success=True,
            arxiv_id=arxiv_id,
            pdf_path=pdf_path,
            latex_path=None,
            metadata=metadata,
            file_size_bytes=pdf_path.stat().st_size,
        )


class DummyDocumentProcessor:
    def __init__(self) -> None:
        self.last_invocation = None

    def process_document(self, pdf_path: Path, latex_path: Path | None, document_id: str | None) -> ProcessingResult:
        self.last_invocation = {
            "pdf_path": pdf_path,
            "latex_path": latex_path,
            "document_id": document_id,
        }

        extraction = ExtractionResult(
            full_text="Example content",
            tables=[],
            equations=[],
            images=[],
            figures=[],
            metadata={"doc_id": document_id},
            latex_source=None,
            has_latex=False,
            extraction_time=0.01,
            extractor_version="test",
        )

        chunk = ChunkWithEmbedding(
            text="Example content",
            embedding=np.zeros(4, dtype=np.float32),
            start_char=0,
            end_char=len("Example content"),
            start_token=0,
            end_token=2,
            chunk_index=0,
            total_chunks=1,
            context_window_used=2,
        )

        return ProcessingResult(
            extraction=extraction,
            chunks=[chunk],
            processing_metadata={"strategy": "late"},
            total_processing_time=0.02,
            extraction_time=0.01,
            chunking_time=0.0,
            embedding_time=0.01,
            success=True,
            errors=[],
            warnings=[],
        )


def build_workflow(tmp_path: Path) -> ArxivSinglePDFWorkflow:
    staging = tmp_path / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    output_dir = tmp_path / "outputs"
    dummy_api = DummyAPIClient(tmp_path)
    dummy_processor = DummyDocumentProcessor()
    config = WorkflowConfig(name="test_workflow", staging_path=staging)
    return ArxivSinglePDFWorkflow(
        config=config,
        api_client=dummy_api,
        document_processor=dummy_processor,
        output_dir=output_dir,
    )


def test_validate_inputs_accepts_url(tmp_path: Path) -> None:
    workflow = build_workflow(tmp_path)
    assert workflow.validate_inputs(arxiv_url="https://arxiv.org/pdf/2310.08560.pdf")


def test_validate_inputs_rejects_missing_identifier(tmp_path: Path) -> None:
    workflow = build_workflow(tmp_path)
    assert not workflow.validate_inputs()


def test_execute_produces_llm_payload(tmp_path: Path) -> None:
    workflow = build_workflow(tmp_path)
    result = workflow.execute(arxiv_id="2310.08560")

    assert result.success
    assert result.metadata["arxiv_id"] == "2310.08560"

    output_path = Path(result.metadata["output_path"])
    with open(output_path, encoding="utf-8") as handle:
        payload = json.load(handle)

    assert payload["arxiv_id"] == "2310.08560"
    assert payload["document"]["chunks"][0]["text"] == "Example content"
