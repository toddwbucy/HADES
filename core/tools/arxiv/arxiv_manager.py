#!/usr/bin/env python3
"""
ArXiv Manager.

ArXiv-specific document management that uses the generic DocumentProcessor
for expensive operations while handling ArXiv-specific metadata, validation,
and storage schemas.

Translates ArXiv-specific concerns (versioned papers, categories, LaTeX sources)
into generic processing tasks while preserving source-specific metadata.
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

from core.database.arango import (
    ArangoMemoryClient,
    resolve_memory_config,
)
from core.processors.document_processor import (
    DocumentProcessor,
    ProcessingConfig,
    ProcessingResult,
)

logger = logging.getLogger(__name__)


@dataclass
class ArXivPaperInfo:
    """
    ArXiv-specific paper information.

    Contains all ArXiv-specific metadata that distinguishes these papers
    from other sources. This represents the unique characteristics that
    make ArXiv papers special in our multi-source architecture.
    """
    arxiv_id: str
    version: str
    pdf_path: Path
    latex_path: Path | None
    title: str
    authors: list[str]
    abstract: str
    categories: list[str]
    submission_date: str
    update_date: str | None
    comments: str | None
    journal_ref: str | None
    doi: str | None

    @property
    def sanitized_id(self) -> str:
        """
        Return a filesystem- and database-safe identifier derived from the ArXiv ID.

        Replaces dots and slashes in the original `arxiv_id` with underscores so the result can be used
        as a filename, database key, or filesystem-safe identifier.
        """
        return self.arxiv_id.replace('.', '_').replace('/', '_')

    @property
    def has_latex(self) -> bool:
        """Check if LaTeX source is available."""
        return self.latex_path is not None and self.latex_path.exists()


class ArXivValidator:
    """
    Validates and parses ArXiv-specific identifiers and paths.

    Handles the complex ArXiv ID formats and directory structures,
    translating them into standardized paths for processing.
    """

    # ArXiv ID patterns
    ARXIV_ID_PATTERN = re.compile(r'^(\d{4})\.(\d{4,5})(v\d+)?$')
    # Old format: archive/YYMMNNN or archive.CC/YYMMNNN where CC is subject class
    # Examples: hep-th/9901001, cs/9901001, math.GT/0309136, cond-mat/9901001
    OLD_ARXIV_ID_PATTERN = re.compile(r'^([a-z][a-z\-]*(?:\.[A-Z]{2})?)\/(\d{7})(v\d+)?$')

    # Base paths for ArXiv data
    PDF_BASE_PATH = Path('/bulk-store/arxiv-data/pdf')
    LATEX_BASE_PATH = Path('/bulk-store/arxiv-data/src')
    METADATA_PATH = Path('/bulk-store/arxiv-data/metadata/arxiv-metadata-oai-snapshot.json')

    @classmethod
    def validate_arxiv_id(cls, arxiv_id: str) -> tuple[bool, str | None]:
        """
        Validate an arXiv identifier.

        Checks whether the given `arxiv_id` matches either the new format (YYMM.NNNNN, optional `vN` suffix allowed) or the legacy format (category/YYMMNNN, optional `vN` suffix allowed). The function ignores any trailing version suffix (`vN`) when validating and returns a tuple (is_valid, error_message) where `error_message` is None on success or a short diagnostic on failure.

        Examples of accepted forms: "2308.12345", "2308.12345v2", "hep-th/9901001", "hep-th/9901001v3".
        """
        # Remove any version suffix for validation
        base_id = re.sub(r'v\d+$', '', arxiv_id)

        # Check new format (YYMM.NNNNN)
        if cls.ARXIV_ID_PATTERN.match(base_id):
            return True, None

        # Check old format (category/YYMMNNN)
        if cls.OLD_ARXIV_ID_PATTERN.match(base_id):
            return True, None

        return False, f"Invalid ArXiv ID format: {arxiv_id}"

    @classmethod
    def get_pdf_path(cls, arxiv_id: str) -> Path:
        """
        Return the filesystem Path to the PDF for the given arXiv identifier.

        Given an arXiv ID (new-style 'YYMM.NNNNN' or old-style 'category/YYMMNNN', with optional version suffix like 'v2'),
        this resolves the expected PDF location under the class PDF_BASE_PATH:
        - new format -> PDF_BASE_PATH / YYMM / 'base_id.pdf'
        - old format -> PDF_BASE_PATH / YYMM / 'category_YYMMNNN.pdf' (slashes replaced with underscores)

        Parameters:
            arxiv_id (str): ArXiv identifier; version suffix (e.g., 'v2') is ignored when computing the path.

        Returns:
            Path: Resolved path to the PDF for the provided arXiv ID.

        Raises:
            ValueError: If the arXiv ID does not match either the new or old recognized formats.
        """
        # Extract base ID without version
        base_id = re.sub(r'v\d+$', '', arxiv_id)

        # Handle new format (YYMM.NNNNN)
        match = cls.ARXIV_ID_PATTERN.match(base_id)
        if match:
            yymm = match.group(1)
            return cls.PDF_BASE_PATH / yymm / f"{base_id}.pdf"

        # Handle old format (category/YYMMNNN)
        match = cls.OLD_ARXIV_ID_PATTERN.match(base_id)
        if match:
            number = match.group(2)
            yymm = number[:4]
            return cls.PDF_BASE_PATH / yymm / f"{base_id.replace('/', '_')}.pdf"

        raise ValueError(f"Cannot determine PDF path for: {arxiv_id}")

    @classmethod
    def get_latex_path(cls, arxiv_id: str) -> Path | None:
        """
        Return the Path to a LaTeX main file for the given arXiv ID if a LaTeX source exists, otherwise None.

        The function ignores any trailing version suffix (e.g., `v2`) and only attempts to resolve LaTeX sources for new-format IDs (YYMM.NNNNN). If a LaTeX directory is found it prefers a main file named `main.tex`, `paper.tex`, or `{base_id}.tex` in that order; if none of those exist it returns the first `*.tex` file found. Returns None when no LaTeX source directory or .tex files are present.
        """
        # Extract base ID without version
        base_id = re.sub(r'v\d+$', '', arxiv_id)

        # Handle new format (YYMM.NNNNN)
        match = cls.ARXIV_ID_PATTERN.match(base_id)
        if match:
            yymm = match.group(1)
            latex_dir = cls.LATEX_BASE_PATH / yymm / base_id
            if latex_dir.exists():
                # Look for main .tex file
                tex_files = list(latex_dir.glob('*.tex'))
                if tex_files:
                    # Prefer main.tex or paper.tex if they exist
                    for preferred in ['main.tex', 'paper.tex', f'{base_id}.tex']:
                        preferred_path = latex_dir / preferred
                        if preferred_path.exists():
                            return preferred_path
                    # Otherwise return first .tex file
                    return tex_files[0]

        return None

    @classmethod
    def parse_arxiv_id(cls, arxiv_id: str) -> dict[str, str]:
        """
        Parse an arXiv identifier into its constituent components.

        Given an arXiv ID (new-style like `YYMM.NNNNN` or old-style like `category/YYMMNNN`) optionally suffixed with a version (`vN`), return a dictionary of extracted fields useful for path resolution and metadata lookup.

        Returned dictionary keys (when present):
        - base_id: the arXiv identifier without any version suffix.
        - version: version suffix (e.g. `v2`). If no version is present, `v1` is returned.
        - format: `'new'` for `YYMM.NNNNN` style IDs or `'old'` for `category/YYMMNNN`.
        - year_month: the `YYMM` portion for new-style IDs or the best-effort year/month fragment for old-style IDs.
        - number: the numeric sequence component of the identifier.
        - category: (old-style only) the subject/category prefix (e.g. `cs.CV`).

        The function does not validate the semantic correctness of fields beyond pattern matching; keys absent in the result indicate the corresponding pattern component was not found.
        """
        components = {}

        # Extract version if present
        version_match = re.search(r'(v\d+)$', arxiv_id)
        if version_match:
            components['version'] = version_match.group(1)
            components['base_id'] = arxiv_id[:-len(version_match.group(1))]
        else:
            components['version'] = 'v1'
            components['base_id'] = arxiv_id

        # Parse base ID format
        match = cls.ARXIV_ID_PATTERN.match(components['base_id'])
        if match:
            components['year_month'] = match.group(1)
            components['number'] = match.group(2)
            components['format'] = 'new'
        else:
            match = cls.OLD_ARXIV_ID_PATTERN.match(components['base_id'])
            if match:
                components['category'] = match.group(1)
                components['number'] = match.group(2)
                components['year_month'] = match.group(2)[:4]
                components['format'] = 'old'

        return components


class ArXivManager:
    """
    ArXiv-specific document management.

    Handles ArXiv metadata, validation, and storage while delegating
    expensive processing to the generic DocumentProcessor. This manager
    preserves all ArXiv-specific characteristics while benefiting from
    shared processing infrastructure.
    """

    def __init__(
        self,
        processing_config: ProcessingConfig | None = None,
        arango_config: dict[str, Any] | None = None,
        max_metadata_entries: int | None = 100000,
        enable_metadata_cache: bool = True,
    ):
        """
        Create a new ArXivManager, wiring the document processor, optional ArangoDB manager, validator, and metadata cache.

        If provided, processing_config is passed to the underlying DocumentProcessor. If arango_config is provided, an ArangoDBManager is created to enable persistent storage; otherwise DB storage is disabled and related methods will be no-ops. The constructor also instantiates an ArXivValidator and attempts to load the in-memory metadata cache from the configured metadata snapshot.

        Parameters:
            processing_config: Optional configuration forwarded to DocumentProcessor (affects parsing/embedding behavior).
            arango_config: Optional ArangoDB connection/configuration; when omitted, database-backed storage is disabled.
            max_metadata_entries: Maximum number of metadata entries to cache in memory (None for unlimited).
                Default is 100000 to prevent OOM with full ArXiv snapshot (~2.8M records).
            enable_metadata_cache: If False, skip preloading metadata cache entirely. Default is True.
        """
        # Initialize generic processor
        self.processor = DocumentProcessor(processing_config)

        # Initialize database client if config provided
        self.db_client: ArangoMemoryClient | None = None
        if arango_config:
            memory_config = resolve_memory_config(**arango_config)
            self.db_client = ArangoMemoryClient(memory_config)

        # Initialize validator
        self.validator = ArXivValidator()

        # Load metadata if available and enabled
        self.metadata_cache: dict[str, Any] = {}
        self._max_metadata_entries = max_metadata_entries
        if enable_metadata_cache:
            self._load_metadata_cache()

        logger.info("Initialized ArXivManager")

    def _load_metadata_cache(self):
        """
        Load the ArXiv metadata snapshot (JSON Lines) into self.metadata_cache.

        Reads the file at ArXivValidator.METADATA_PATH, parses each non-empty line as JSON,
        and stores each paper object keyed by its 'id' field into self.metadata_cache.
        Blank lines are ignored. If the file is missing the method does nothing.
        Loading stops after max_metadata_entries (if set) to prevent OOM.
        Any exceptions raised while reading or parsing the snapshot are caught and result
        in a logged warning; the exception is not propagated.
        """
        metadata_path = ArXivValidator.METADATA_PATH
        if metadata_path.exists():
            try:
                entries_loaded = 0
                truncated = False
                with open(metadata_path, encoding='utf-8', errors='replace') as f:
                    # ArXiv metadata is typically in JSON Lines format
                    for line in f:
                        if line.strip():
                            # Check memory bound before loading more entries
                            if self._max_metadata_entries is not None and entries_loaded >= self._max_metadata_entries:
                                truncated = True
                                break
                            paper = json.loads(line)
                            arxiv_id = paper.get('id', '')
                            if arxiv_id:
                                self.metadata_cache[arxiv_id] = paper
                                entries_loaded += 1

                if truncated:
                    logger.info(
                        f"Loaded metadata for {len(self.metadata_cache)} papers "
                        f"(truncated at {self._max_metadata_entries} entries to prevent OOM)"
                    )
                else:
                    logger.info(f"Loaded metadata for {len(self.metadata_cache)} papers")
            except Exception as e:
                logger.warning(f"Failed to load metadata cache: {e}")

    def get_paper_info(self, arxiv_id: str) -> ArXivPaperInfo:
        """
        Return a populated ArXivPaperInfo for the given ArXiv identifier.

        Validates and parses the provided `arxiv_id`, resolves filesystem paths for the PDF
        (and optional LaTeX source), and combines those with metadata loaded from the
        in-memory metadata cache to construct an ArXivPaperInfo instance.

        Parameters:
            arxiv_id: ArXiv identifier in either new format (YYMM.NNNNN[vN]) or old format
                (category/YYMMNNN[vN]). The function strips/uses any version suffix when
                appropriate.

        Returns:
            ArXivPaperInfo populated with filesystem paths, version, title, authors, abstract,
            categories, submission/update dates and optional fields (comments, journal_ref, doi).

        Raises:
            ValueError: if `arxiv_id` is not a valid ArXiv identifier.
            FileNotFoundError: if the resolved PDF file does not exist.
        """
        # Validate ID
        is_valid, error = self.validator.validate_arxiv_id(arxiv_id)
        if not is_valid:
            raise ValueError(error)

        # Parse ID components
        id_components = self.validator.parse_arxiv_id(arxiv_id)

        # Get paths
        pdf_path = self.validator.get_pdf_path(arxiv_id)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        latex_path = self.validator.get_latex_path(arxiv_id)

        # Get metadata from cache or use defaults
        metadata = self.metadata_cache.get(id_components['base_id'], {})

        raw_authors = metadata.get('authors', [])
        authors = (
            raw_authors
            if isinstance(raw_authors, list)
            else [a.strip() for a in re.split(r'[;,]\s*', raw_authors) if a.strip()]
        )

        raw_categories = metadata.get('categories', [])
        categories = (
            raw_categories
            if isinstance(raw_categories, list)
            else raw_categories.split()
        )

        return ArXivPaperInfo(
            arxiv_id=arxiv_id,
            version=id_components['version'],
            pdf_path=pdf_path,
            latex_path=latex_path,
            title=metadata.get('title', f'ArXiv Paper {arxiv_id}'),
            authors=authors,
            abstract=metadata.get('abstract', ''),
            categories=categories,
            submission_date=metadata.get('created', ''),
            update_date=metadata.get('updated'),
            comments=metadata.get('comments'),
            journal_ref=metadata.get('journal_ref'),
            doi=metadata.get('doi')
        )

    async def process_arxiv_paper(
        self,
        arxiv_id: str,
        store_in_db: bool = True
    ) -> ProcessingResult:
        """
        Process a single arXiv paper: run the generic document processor and attach arXiv-specific metadata.

        Retrieves arXiv-specific paths and metadata for the given arXiv_id, invokes the shared DocumentProcessor on the paper's PDF (and optional LaTeX), augments the returned ProcessingResult.processing_metadata with arXiv fields, and optionally persists the result to ArangoDB.

        Parameters:
            arxiv_id (str): ArXiv identifier (new or old format); invalid IDs raise ValueError.
            store_in_db (bool): If True and a DB manager is configured, persist processing results to the arXiv-specific collections.

        Returns:
            ProcessingResult: Result from the generic processor with added keys: 'source', 'arxiv_id', 'version', 'categories', 'has_latex_source', and 'submission_date'.

        Raises:
            ValueError: If the arXiv_id is invalid.
            FileNotFoundError: If the referenced PDF cannot be found.
        """
        logger.info(f"Processing ArXiv paper: {arxiv_id}")

        # Step 1: Get ArXiv-specific information
        paper_info = self.get_paper_info(arxiv_id)

        # Step 2: Use generic processor for expensive operations
        loop = asyncio.get_running_loop()
        process_fn = partial(
            self.processor.process_document,
            pdf_path=paper_info.pdf_path,
            latex_path=paper_info.latex_path,
            document_id=arxiv_id,
        )
        processing_result = await loop.run_in_executor(None, process_fn)

        # Step 3: Attach ArXiv-specific metadata
        processing_result.processing_metadata.update({
            'source': 'arxiv',
            'arxiv_id': arxiv_id,
            'version': paper_info.version,
            'categories': paper_info.categories,
            'has_latex_source': paper_info.has_latex,
            'submission_date': paper_info.submission_date
        })

        # Step 4: Store in ArXiv-specific collections if requested
        if store_in_db and self.db_client and processing_result.success:
            store_fn = partial(self._store_arxiv_result_sync, paper_info, processing_result)
            await loop.run_in_executor(None, store_fn)

        return processing_result

    def _store_arxiv_result_sync(
        self,
        paper_info: ArXivPaperInfo,
        result: ProcessingResult
    ):
        """
        Persist ArXiv processing results and extracted artifacts to the configured ArangoDB collections.

        If no database client is configured, the call is a no-op (it logs a warning and returns). When a client is present, this inserts:
        - a main document into the `arxiv_metadata` collection (keyed by paper_info.sanitized_id) containing ArXiv metadata and processing metrics;
        - one document per text chunk with embeddings into `arxiv_abstract_embeddings`;
        - an optional `arxiv_structures` document containing tables, equations, images, and figures when any are present.

        Errors raised by the DB manager or during document insertion are logged and re-raised.
        """
        if not self.db_client:
            logger.warning("No database client configured, skipping storage")
            return

        try:
            # Prepare ArXiv-specific document
            arxiv_doc = {
                '_key': paper_info.sanitized_id,
                'arxiv_id': paper_info.arxiv_id,
                'version': paper_info.version,
                'title': paper_info.title,
                'authors': paper_info.authors,
                'abstract': paper_info.abstract,
                'categories': paper_info.categories,
                'submission_date': paper_info.submission_date,
                'update_date': paper_info.update_date,
                'comments': paper_info.comments,
                'journal_ref': paper_info.journal_ref,
                'doi': paper_info.doi,
                'has_latex': paper_info.has_latex,
                'processing_timestamp': datetime.now(UTC).isoformat(),
                'processing_metadata': result.processing_metadata,
                'num_chunks': len(result.chunks),
                'num_tables': len(result.extraction.tables),
                'num_equations': len(result.extraction.equations),
                'num_images': len(result.extraction.images),
                'status': 'PROCESSED'
            }

            now_iso = datetime.now(UTC).isoformat()

            chunk_docs: list[dict[str, Any]] = []
            embedding_docs: list[dict[str, Any]] = []
            for chunk in result.chunks:
                chunk_id = f"{paper_info.sanitized_id}_chunk_{chunk.chunk_index}"
                chunk_docs.append({
                    '_key': chunk_id,
                    'arxiv_id': paper_info.arxiv_id,
                    'document_id': paper_info.arxiv_id,
                    'paper_key': paper_info.sanitized_id,
                    'chunk_index': chunk.chunk_index,
                    'total_chunks': chunk.total_chunks,
                    'text': chunk.text,
                    'start_char': chunk.start_char,
                    'end_char': chunk.end_char,
                    'context_window_used': chunk.context_window_used,
                    'created_at': now_iso,
                })

                embedding_docs.append({
                    '_key': f"{chunk_id}_emb",
                    'chunk_id': chunk_id,
                    'arxiv_id': paper_info.arxiv_id,
                    'document_id': paper_info.arxiv_id,
                    'paper_key': paper_info.sanitized_id,
                    'embedding': chunk.embedding.tolist(),
                    'embedding_dim': int(chunk.embedding.shape[0]) if hasattr(chunk.embedding, 'shape') else len(chunk.embedding),
                    'model': getattr(self.processor, 'embedding_model', 'jinaai/jina-embeddings-v4'),
                    'created_at': now_iso,
                })

            structures_docs: list[dict[str, Any]] = []
            if result.extraction.tables or result.extraction.equations or result.extraction.images:
                structures_docs.append({
                    '_key': paper_info.sanitized_id,
                    'arxiv_id': paper_info.arxiv_id,
                    'document_id': paper_info.arxiv_id,
                    'tables': result.extraction.tables,
                    'equations': result.extraction.equations,
                    'images': result.extraction.images,
                    'figures': result.extraction.figures,
                    'updated_at': now_iso,
                })

            # Use Stream Transaction for atomic inserts
            write_collections = [
                'arxiv_metadata',
                'arxiv_abstract_chunks',
                'arxiv_abstract_embeddings',
                'arxiv_structures'
            ]

            transaction_id = self.db_client.begin_transaction(write=write_collections)

            try:
                # Insert documents within the transaction
                chunks_inserted = 0
                embeddings_inserted = 0
                structures_inserted = 0

                if arxiv_doc:
                    self.db_client.insert_in_transaction(
                        'arxiv_metadata', [arxiv_doc], transaction_id
                    )

                if chunk_docs:
                    self.db_client.insert_in_transaction(
                        'arxiv_abstract_chunks', chunk_docs, transaction_id
                    )
                    chunks_inserted = len(chunk_docs)

                if embedding_docs:
                    self.db_client.insert_in_transaction(
                        'arxiv_abstract_embeddings', embedding_docs, transaction_id
                    )
                    embeddings_inserted = len(embedding_docs)

                if structures_docs:
                    self.db_client.insert_in_transaction(
                        'arxiv_structures', structures_docs, transaction_id
                    )
                    structures_inserted = len(structures_docs)

                # Commit the transaction
                self.db_client.commit_transaction(transaction_id)

                logger.info(
                    "Stored ArXiv paper %s (chunks=%d, embeddings=%d, structures=%d)",
                    paper_info.arxiv_id,
                    chunks_inserted,
                    embeddings_inserted,
                    structures_inserted,
                )

            except Exception:
                # Abort transaction on any error
                try:
                    self.db_client.abort_transaction(transaction_id)
                except Exception as abort_exc:
                    logger.warning("Failed to abort transaction: %s", abort_exc)
                raise

        except Exception:
            logger.exception("Failed to store ArXiv paper %s", paper_info.arxiv_id)
            raise

    async def process_batch(
        self,
        arxiv_ids: list[str],
        store_in_db: bool = True
    ) -> list[ProcessingResult]:
        """
        Process a list of ArXiv papers and return their processing results.

        Processes each arXiv ID sequentially by calling process_arxiv_paper. If processing a specific ID raises an exception, this method logs the error and appends a synthesized failed ProcessingResult (success=False) containing the arXiv ID and the error message so the returned list preserves an entry for every requested ID.

        Parameters:
            arxiv_ids (List[str]): ArXiv IDs to process.
            store_in_db (bool): If True (default), successful results will be stored in the configured database when available.

        Returns:
            List[ProcessingResult]: A list of ProcessingResult objects in the same order as arxiv_ids. Entries for IDs that failed during processing are returned as failed ProcessingResult instances containing the error details.
        """
        async def _process_one(aid: str) -> ProcessingResult:
            try:
                return await self.process_arxiv_paper(aid, store_in_db)
            except Exception as exc:  # pragma: no cover - surfaced to caller
                logger.exception("Failed to process %s", aid)
                from core.processors.document_processor import ExtractionResult, ProcessingResult
                return ProcessingResult(
                    extraction=ExtractionResult(full_text=""),
                    chunks=[],
                    processing_metadata={'arxiv_id': aid, 'error': str(exc)},
                    total_processing_time=0,
                    extraction_time=0,
                    chunking_time=0,
                    embedding_time=0,
                    success=False,
                    errors=[str(exc)]
                )

        sem = asyncio.Semaphore(min(8, max(1, len(arxiv_ids))))

        async def _guarded(aid: str) -> ProcessingResult:
            async with sem:
                return await _process_one(aid)

        return await asyncio.gather(*(_guarded(aid) for aid in arxiv_ids))
