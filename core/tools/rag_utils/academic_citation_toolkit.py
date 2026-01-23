#!/usr/bin/env python3
"""
Academic Citation Toolkit.

SOURCE-AGNOSTIC citation and bibliography extraction toolkit.

This toolkit works with ANY academic paper corpus:
- ArXiv papers
- SSRN papers
- Harvard Law Library
- PubMed articles
- Any academic paper with citations and bibliography

The toolkit is completely independent of the source and can be used
as a utility for building citation networks from any academic corpus.
"""

import json
import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BibliographyEntry:
    """
    Represents a single bibliography entry from any academic paper.

    SOURCE-AGNOSTIC: Works regardless of whether the paper comes from
    ArXiv, SSRN, law library, PubMed, or any other academic source.
    """
    entry_number: str | None  # [1], [2], etc.
    raw_text: str
    authors: list[str]
    title: str | None
    venue: str | None  # Journal/conference
    year: int | None
    arxiv_id: str | None
    doi: str | None
    pmid: str | None  # PubMed ID
    ssrn_id: str | None  # SSRN ID
    url: str | None
    source_paper_id: str
    confidence: float

@dataclass
class InTextCitation:
    """
    Represents an in-text citation that points to a bibliography entry.

    SOURCE-AGNOSTIC: Works with any academic citation format.
    """
    raw_text: str
    citation_type: str  # 'numeric', 'author_year', 'hybrid'
    start_pos: int
    end_pos: int
    context: str
    bibliography_ref: str | None  # Points to bibliography entry
    source_paper_id: str
    confidence: float

class DocumentProvider(ABC):
    """
    Abstract interface for providing document text.

    This allows the citation toolkit to work with ANY document source:
    - ArangoDB (our current ArXiv setup)
    - File system (PDF/text files)
    - Database (PostgreSQL, MongoDB, etc.)
    - APIs (SSRN, PubMed, etc.)
    - Web scraping
    """

    @abstractmethod
    def get_document_text(self, document_id: str) -> str | None:
        """
        Return the full text of a document identified by document_id, or None if the document is not found or cannot be retrieved.

        Implementations should accept a provider-specific document identifier (e.g., paper id, filename, or database key) and return the document's complete text as a single string. If the document does not exist or an error occurs while fetching it, return None.
        """
        pass

    @abstractmethod
    def get_document_chunks(self, document_id: str) -> list[str]:
        """
        Return the document's text split into sequential text chunks (paragraph-like segments).

        The implementation is expected to return a list of non-empty, stripped string chunks representing the document's text (e.g., paragraph or section fragments) in their original order. If the document cannot be found or an error occurs, implementations should return an empty list.
        """
        pass

class ArangoDocumentProvider(DocumentProvider):
    """
    Document provider for ArangoDB (our current ArXiv setup).
    """

    def __init__(self, arango_client, db_name: str = 'academy_store', username: str | None = None):
        """
        Initialize the ArangoDocumentProvider and establish a connection to the ArangoDB database.

        Parameters:
            db_name (str): Name of the ArangoDB database to connect to (default: 'academy_store').
            username (str | None): Optional username to authenticate with; if omitted, the `ARANGO_USERNAME`
                environment variable is used (defaults to 'root' when that variable is not set).

        Raises:
            ValueError: If the `ARANGO_PASSWORD` environment variable is not set.

        Side effects:
            - Stores the provided Arango client on `self.client`.
            - Opens and stores a database connection on `self.db`.
        """
        self.client = arango_client
        username = username or os.getenv('ARANGO_USERNAME', 'root')
        password = os.getenv('ARANGO_PASSWORD')
        if not password:
            raise ValueError("ARANGO_PASSWORD environment variable required")
        self.db = arango_client.db(db_name, username=username, password=password)

    def get_document_text(self, document_id: str) -> str | None:
        """Get full document text by combining all chunks."""
        chunks = self.get_document_chunks(document_id)
        return ' '.join(chunks) if chunks else None

    def get_document_chunks(self, document_id: str) -> list[str]:
        """
        Retrieve ordered text chunks for a paper from the ArangoDB `arxiv_abstract_chunks` collection.

        Given a paper identifier, queries the `arxiv_abstract_chunks` collection for documents with matching
        `paper_id`, sorts results by `chunk_index` ascending, and returns the list of chunk texts.
        On any error (including query failure) an empty list is returned.
        """
        try:
            cursor = self.db.aql.execute("""
            FOR chunk IN arxiv_abstract_chunks
                FILTER chunk.paper_id == @paper_id
                SORT chunk.chunk_index ASC
                RETURN chunk.text
            """, bind_vars={'paper_id': document_id})

            return list(cursor)
        except Exception as e:
            logger.error(f"Error getting chunks for {document_id}: {e}")
            return []

class FileSystemDocumentProvider(DocumentProvider):
    """
    Document provider for local files (PDFs, text files).
    """

    def __init__(self, base_path: str):
        """
        Initialize the FileSystemDocumentProvider.

        Parameters:
            base_path (str): Filesystem directory containing text documents (each document expected as `<document_id>.txt`).
        """
        self.base_path = base_path

    def get_document_text(self, document_id: str) -> str | None:
        """
        Return the full text of a document stored as "<document_id>.txt" under the provider's base path.

        If the file is found and readable, returns its contents as a UTF-8 string.
        If the file cannot be opened or read, logs the error and returns None.
        Rejects document_id values that would escape the base_path via path traversal.
        """
        from pathlib import Path

        try:
            # Construct path safely using pathlib
            base_path = Path(self.base_path).resolve()
            file_path = (base_path / f"{document_id}.txt").resolve()

            # Prevent path traversal attacks - ensure file is within base_path
            if not str(file_path).startswith(str(base_path)):
                logger.error(f"Path traversal attempt detected for document_id: {document_id}")
                return None

            with open(file_path, encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading {document_id}: {e}")
            return None

    def get_document_chunks(self, document_id: str) -> list[str]:
        """
        Return the document split into paragraph chunks.

        Each chunk is produced by splitting the document text on double newlines, stripping surrounding whitespace, and discarding empty paragraphs. If the document text is unavailable or empty, returns an empty list.
        """
        text = self.get_document_text(document_id)
        if not text:
            return []

        # Simple paragraph-based chunking
        return [p.strip() for p in text.split('\n\n') if p.strip()]

class CitationStorage(ABC):
    """
    Abstract interface for storing citation data.

    This allows the toolkit to store results in any system:
    - ArangoDB (our current setup)
    - PostgreSQL
    - JSON files
    - CSV files
    - Any other storage system
    """

    @abstractmethod
    def store_bibliography_entries(self, entries: list[BibliographyEntry]) -> bool:
        """
        Persist a list of BibliographyEntry records to the configured backend.

        Implementations should attempt to store all provided entries and return True if the operation completed successfully (entries persisted or already present), or False if the storage failed.

        Parameters:
            entries (List[BibliographyEntry]): Bibliography entries to persist; may be empty.

        Returns:
            bool: True on successful storage, False on failure.
        """
        pass

    @abstractmethod
    def store_citations(self, citations: list[InTextCitation]) -> bool:
        """
        Persist a list of in-text citation records to the storage backend.

        Parameters:
            citations (List[InTextCitation]): List of in-text citation dataclass instances to persist.

        Returns:
            bool: True if all citations were stored successfully, False on any failure.
        """
        pass

class ArangoCitationStorage(CitationStorage):
    """Citation storage for ArangoDB."""

    def __init__(self, arango_client, db_name: str = 'academy_store', username: str | None = None):
        """
        Initialize the ArangoCitationStorage and establish a connection to the ArangoDB database.

        Parameters:
            arango_client: ArangoDB client instance for database operations.
            db_name (str): Name of the ArangoDB database to connect to (default: 'academy_store').
            username (str | None): Optional username to authenticate with; if omitted, the `ARANGO_USERNAME`
                environment variable is used (defaults to 'root' when that variable is not set).

        Raises:
            ValueError: If the `ARANGO_PASSWORD` environment variable is not set.

        Side effects:
            - Stores the provided Arango client on `self.client`.
            - Opens and stores a database connection on `self.db`.
        """
        self.client = arango_client
        username = username or os.getenv('ARANGO_USERNAME', 'root')
        password = os.getenv('ARANGO_PASSWORD')
        if not password:
            raise ValueError("ARANGO_PASSWORD environment variable required")
        self.db = arango_client.db(db_name, username=username, password=password)

    def store_bibliography_entries(self, entries: list[BibliographyEntry]) -> bool:
        """
        Persist a list of BibliographyEntry objects into the ArangoDB `bibliography_entries` collection.

        If `entries` is empty this is a no-op and returns True. Each entry is stored as a document with a generated `_key` of the form
        `{source_paper_id}_{entry_number or 'unknown'}`. Documents are inserted; on unique-key conflicts the existing document is updated. Returns True on successful completion, or False if an error occurs while accessing or writing to the database.
        """
        if not entries:
            return True

        try:
            bibliography_collection = self.db.collection('bibliography_entries')

            for entry in entries:
                doc = {
                    '_key': f"{entry.source_paper_id}_{entry.entry_number or 'unknown'}",
                    'source_paper_id': entry.source_paper_id,
                    'entry_number': entry.entry_number,
                    'raw_text': entry.raw_text,
                    'authors': entry.authors,
                    'title': entry.title,
                    'venue': entry.venue,
                    'year': entry.year,
                    'arxiv_id': entry.arxiv_id,
                    'doi': entry.doi,
                    'pmid': entry.pmid,
                    'ssrn_id': entry.ssrn_id,
                    'url': entry.url,
                    'confidence': entry.confidence
                }

                try:
                    bibliography_collection.insert(doc)
                except Exception as e:
                    if "unique constraint violated" in str(e).lower():
                        bibliography_collection.update({'_key': doc['_key']}, doc)
                    else:
                        raise e

            return True
        except Exception as e:
            logger.error(f"Error storing bibliography entries: {e}")
            return False

    def store_citations(self, citations: list[InTextCitation]) -> bool:
        """
        Store a list of in-text citation records in ArangoDB.

        This method is intended to persist each InTextCitation (typically by constructing
        a document keyed from the citation's source_paper_id and citation span or id)
        and to upsert on unique-key conflicts. It returns True on successful storage
        and False on failure.

        Parameters:
            citations (List[InTextCitation]): List of in-text citation dataclass instances to store.

        Returns:
            bool: True if storage succeeded, False on error.

        Raises:
            NotImplementedError: This method is not yet implemented.
        """
        raise NotImplementedError(
            "store_citations is not yet implemented. "
            "This method must accept List[InTextCitation] and persist/upsert records "
            "to the 'citations' collection, similar to store_bibliography."
        )

class JSONCitationStorage(CitationStorage):
    """Citation storage to JSON files."""

    def __init__(self, output_dir: str):
        """
        Initialize the JSONCitationStorage.

        Creates the output directory if it does not already exist.

        Parameters:
            output_dir (str): Path to the directory where JSON files (e.g., bibliography.json and citations.json) will be written.
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def store_bibliography_entries(self, entries: list[BibliographyEntry]) -> bool:
        """
        Write a list of BibliographyEntry objects to a JSON file named `bibliography.json` under the instance's output directory.

        The entries are serialized (by their dataclass attributes) and written with indentation; non-JSON-serializable values are converted using `str`. Returns True on successful write, False if an error occurs.
        """
        import json

        try:
            data = [entry.__dict__ for entry in entries]
            with open(f"{self.output_dir}/bibliography.json", 'w') as f:
                json.dump(data, f, indent=2, default=str)
            return True
        except Exception as e:
            logger.error(f"Error storing to JSON: {e}")
            return False

    def store_citations(self, citations: list[InTextCitation]) -> bool:
        """
        Serialize a list of InTextCitation objects and write them to <output_dir>/citations.json.

        Parameters:
            citations (List[InTextCitation]): List of in-text citation dataclass instances; each will be converted with dataclasses.asdict() before serialization.

        Returns:
            bool: True on successful write, False if an error occurred while writing the file.
        """
        try:
            data = [asdict(citation) for citation in citations]
            with open(f"{self.output_dir}/citations.json", 'w') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Error storing citations to JSON: {e}")
            return False

class UniversalBibliographyExtractor:
    """
    SOURCE-AGNOSTIC bibliography extractor that works with any academic corpus.

    This class can extract bibliographies from papers regardless of source:
    ArXiv, SSRN, Harvard Law, PubMed, or any other academic paper collection.
    """

    def __init__(self, document_provider: DocumentProvider):
        """
        Initialize the UniversalBibliographyExtractor with a document provider.

        The provider is used to fetch full paper text and text chunks for bibliography section detection and entry parsing.
        """
        self.document_provider = document_provider

    def extract_bibliography_section(self, paper_id: str) -> str | None:
        """
        Locate and return the bibliography/references section text for a given paper, if present.

        Attempts multiple, source-agnostic strategies to find a references block:
        1. Search for explicit section headers (e.g., "References", "Bibliography", "Works Cited") and return the following content when substantial.
        2. Search near the end of the document for numbered reference lists (e.g., entries starting with `[n]`).
        3. Search near the end of the document for author–year style reference blocks (repeated "Lastname, YYYY" lines).

        Returns:
            The raw text of the detected bibliography/references section, or None if no suitable section is found or the document text is unavailable.
        """
        try:
            full_text = self.document_provider.get_document_text(paper_id)
            if not full_text:
                return None

            # Strategy 1: Look for explicit "References" section
            references_patterns = [
                r'\b(References|Bibliography|REFERENCES|BIBLIOGRAPHY|Works Cited|Literature Cited)\b.*?(?=\n\n[A-Z][a-z]+|\Z)',
                r'\b(References|Bibliography)\b(.*?)(?=\n\n|\Z)',
                r'## References(.*?)(?=\n##|\Z)',
                r'# References(.*?)(?=\n#|\Z)'
            ]

            for pattern in references_patterns:
                match = re.search(pattern, full_text, re.DOTALL | re.IGNORECASE)
                if match and len(match.group(0)) > 50:  # Must have substantial content
                    logger.info(f"Found references section for {paper_id} (pattern: {pattern[:20]}...)")
                    return match.group(0)

            # Strategy 2: Look for numbered reference pattern at end
            numbered_refs = re.search(
                r'(\[\d+\].*?)(?=\n\n[A-Z][a-z]+|\Z)',
                full_text[-8000:],  # Look in last 8000 chars
                re.DOTALL
            )

            if numbered_refs and len(numbered_refs.group(0)) > 200:
                logger.info(f"Found numbered references for {paper_id}")
                return numbered_refs.group(0)

            # Strategy 3: Look for author-year style references
            author_year_refs = re.search(
                r'((?:[A-Z][a-z]+(?:\s+et\s+al\.?)?,\s+\d{4}.*?\n){5,})',
                full_text[-5000:],  # Look in last 5000 chars
                re.DOTALL
            )

            if author_year_refs:
                logger.info(f"Found author-year references for {paper_id}")
                return author_year_refs.group(0)

            logger.warning(f"No references section found for {paper_id}")
            return None

        except Exception as e:
            logger.error(f"Error extracting bibliography for {paper_id}: {e}")
            return None

    def parse_bibliography_entries(self, bibliography_text: str, paper_id: str) -> list[BibliographyEntry]:
        """
        Parse a bibliography section into a list of BibliographyEntry objects.

        Attempts multiple tolerant strategies to handle common bibliography formats (numbered entries like "[1]", paragraph-separated entries, and multi-line author-year or prose-style entries). For each detected entry it calls _parse_single_entry; entries with very short content are skipped. Entry numbering is preserved from numbered lists when present or generated sequentially for paragraph/split strategies. On error the function returns an empty list.

        Parameters:
            bibliography_text (str): The raw text of a paper's bibliography/references section (as extracted by extract_bibliography_section).
            paper_id (str): Identifier of the source paper; assigned to each returned BibliographyEntry as source_paper_id.

        Returns:
            List[BibliographyEntry]: Parsed bibliography entries (may be empty). Entries are filtered to remove trivial/too-short items; ordering follows the order found in the input.
        """
        entries = []

        try:
            # Remove the header line if present
            text = re.sub(r'^(References|Bibliography|REFERENCES|BIBLIOGRAPHY|Works Cited|Literature Cited)\s*\n?', '', bibliography_text, flags=re.IGNORECASE)

            # Strategy 1: Numbered entries [1], [2], etc.
            numbered_entries = re.findall(
                r'\[(\d+)\]\s*(.*?)(?=\[\d+\]|\Z)',
                text,
                re.DOTALL
            )

            if numbered_entries:
                logger.info(f"Found {len(numbered_entries)} numbered bibliography entries")
                for num, entry_text in numbered_entries:
                    entry = self._parse_single_entry(entry_text.strip(), paper_id, entry_number=num)
                    if entry and len(entry.raw_text.strip()) > 20:  # Must have substantial content
                        entries.append(entry)
                return entries

            # Strategy 2: Split by double newlines (paragraph-separated)
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip() and len(p.strip()) > 30]

            for i, paragraph in enumerate(paragraphs, 1):
                entry = self._parse_single_entry(paragraph, paper_id, entry_number=str(i))
                if entry:
                    entries.append(entry)

            # Strategy 3: Split by single newlines but group multi-line entries
            if not entries:
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                current_entry = ""
                entry_num = 1

                for line in lines:
                    # Start of new entry (starts with capital letter or number)
                    if re.match(r'^([A-Z]|\d+\.)', line) and len(current_entry) > 50:
                        entry = self._parse_single_entry(current_entry, paper_id, entry_number=str(entry_num))
                        if entry:
                            entries.append(entry)
                        current_entry = line
                        entry_num += 1
                    else:
                        current_entry += " " + line

                # Don't forget the last entry
                if len(current_entry) > 50:
                    entry = self._parse_single_entry(current_entry, paper_id, entry_number=str(entry_num))
                    if entry:
                        entries.append(entry)

            logger.info(f"Successfully parsed {len(entries)} bibliography entries for {paper_id}")
            return entries

        except Exception as e:
            logger.error(f"Error parsing bibliography entries for {paper_id}: {e}")
            return []

    def _parse_single_entry(self, entry_text: str, paper_id: str, entry_number: str | None = None) -> BibliographyEntry | None:
        """
        Parse a single bibliography entry string into a BibliographyEntry dataclass.

        This attempts to extract common bibliographic fields (arXiv ID, DOI, PMID, SSRN ID, year,
        title, authors, venue) from a free-form entry line or paragraph. A numeric confidence
        score (0.0–1.0) is computed based on which fields were found to indicate parsing quality.

        Parameters:
            entry_text: The raw bibliography entry text to parse.
            paper_id: The identifier of the source paper; stored as `source_paper_id` on the result.
            entry_number: Optional original entry number or label from the bibliography (if available).

        Returns:
            A BibliographyEntry populated with any extracted fields and a confidence score,
            or None if the input is too short to parse or an error occurs.
        """
        if len(entry_text.strip()) < 20:
            return None

        try:
            # Extract ArXiv ID
            arxiv_match = re.search(r'arXiv:(\d{4}\.\d{4,5})', entry_text, re.IGNORECASE)
            arxiv_id = arxiv_match.group(1) if arxiv_match else None

            # Extract DOI (DOI format: 10.xxxxx/suffix)
            doi_match = re.search(r'doi:?\s*(10\.\d+/[^\s,]+)', entry_text, re.IGNORECASE)
            doi = doi_match.group(1) if doi_match else None

            # Extract PubMed ID
            pmid_match = re.search(r'PMID:?\s*(\d+)', entry_text, re.IGNORECASE)
            pmid = pmid_match.group(1) if pmid_match else None

            # Extract SSRN ID
            ssrn_match = re.search(r'SSRN[:\s]*(\d+)', entry_text, re.IGNORECASE)
            ssrn_id = ssrn_match.group(1) if ssrn_match else None

            # Extract year
            year_match = re.search(r'\b(19|20)\d{2}\b', entry_text)
            year = int(year_match.group(0)) if year_match else None

            # Extract title (text in quotes or italics)
            title_patterns = [
                r'["""]([^"""]{15,200})["""]',  # Quoted titles
                r'_([^_]{15,200})_',  # Italicized titles (markdown)
                r'\*([^*]{15,200})\*',  # Bold titles (markdown)
                r'(?:^|\. )([A-Z][^.]{15,150})\.',  # Title after authors, before period
            ]

            title = None
            for pattern in title_patterns:
                title_match = re.search(pattern, entry_text)
                if title_match:
                    title = title_match.group(1).strip()
                    break

            # Extract authors
            authors = []
            author_patterns = [
                r'^([^.]+(?:[A-Z]\.[^.]*\.)+)',  # "Last, F., Last2, F2."
                r'^([A-Z][a-z]+(?:\s+[A-Z]\.[^,]*,\s*)*[A-Z][a-z]+)',  # "Smith, J., Jones, P."
                r'([A-Z][a-z]+\s+et\s+al\.?)',  # "Smith et al."
            ]

            for pattern in author_patterns:
                author_match = re.search(pattern, entry_text)
                if author_match:
                    author_text = author_match.group(1)
                    # Simple split by commas
                    potential_authors = [a.strip() for a in author_text.split(',')]
                    authors = [a for a in potential_authors if len(a) > 2 and not a.isdigit()][:5]
                    break

            # Extract venue
            venue = None
            venue_patterns = [
                r'In\s+([A-Z][^,\n]{10,50})',  # "In Conference Name"
                r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+\d{4}',  # "Journal Name 2020"
                r'Proceedings\s+of\s+([^,\n]{10,50})',  # "Proceedings of ..."
            ]

            for pattern in venue_patterns:
                venue_match = re.search(pattern, entry_text)
                if venue_match:
                    venue = venue_match.group(1).strip()
                    break

            # Calculate confidence
            confidence = 0.3  # Base confidence
            if arxiv_id or doi or pmid or ssrn_id:
                confidence += 0.4
            if title and len(title) > 10:
                confidence += 0.2
            if authors:
                confidence += 0.2
            if year:
                confidence += 0.1
            if venue:
                confidence += 0.1

            confidence = min(1.0, confidence)

            return BibliographyEntry(
                entry_number=entry_number,
                raw_text=entry_text,
                authors=authors,
                title=title,
                venue=venue,
                year=year,
                arxiv_id=arxiv_id,
                doi=doi,
                pmid=pmid,
                ssrn_id=ssrn_id,
                url=None,
                source_paper_id=paper_id,
                confidence=confidence
            )

        except Exception as e:
            logger.error(f"Error parsing bibliography entry: {e}")
            return None

    def extract_paper_bibliography(self, paper_id: str) -> list[BibliographyEntry]:
        """
        Extract the bibliography for a single paper and return parsed entries.

        Given a paper ID, locate the paper's bibliography/references section and parse it into
        a list of BibliographyEntry objects. If no bibliography section can be found or an error
        occurs during extraction, an empty list is returned.

        Parameters:
            paper_id (str): Identifier of the source paper (e.g., arXiv ID or local document ID).

        Returns:
            List[BibliographyEntry]: Parsed bibliography entries (may be empty if none found).
        """
        logger.info(f"Extracting bibliography for paper {paper_id}")

        # Extract bibliography section
        bibliography_text = self.extract_bibliography_section(paper_id)
        if not bibliography_text:
            return []

        # Parse entries
        entries = self.parse_bibliography_entries(bibliography_text, paper_id)

        logger.info(f"Successfully extracted {len(entries)} bibliography entries for {paper_id}")
        return entries

class UniversalCitationExtractor:
    """
    SOURCE-AGNOSTIC in-text citation extractor.

    Extracts citations from any academic paper and maps them to
    bibliography entries regardless of the source.
    """

    def __init__(self, document_provider: DocumentProvider):
        """
        Initialize the UniversalCitationExtractor with a document provider.

        The provider is used to fetch full paper text and text chunks for
        in-text citation detection and bibliography entry matching.
        """
        self.document_provider = document_provider

    def extract_citations(self, paper_id: str, bibliography_entries: list[BibliographyEntry]) -> list[InTextCitation]:
        """
        Extract in-text citations from a paper and attempt to link each citation to a parsed bibliography entry.

        This method scans the full text of the paper identified by `paper_id` (via the provider attached to the extractor) to locate in-text citation tokens (e.g., numbered forms like “[1]”, parenthetical numeric lists, and author–year forms like “(Smith, 2020)”). For each detected citation it produces an InTextCitation containing the citation text, its span (start_pos, end_pos), surrounding context, a best-effort `bibliography_ref` that refers to the matched BibliographyEntry (if resolvable), and a confidence score reflecting match quality.

        Parameters:
            paper_id (str): Identifier of the source paper whose text will be scanned for citations.
            bibliography_entries (List[BibliographyEntry]): Parsed bibliography entries for `paper_id` used to resolve and disambiguate in-text mentions.

        Returns:
            List[InTextCitation]: A list of discovered in-text citations. If no citations are found, returns an empty list. The method does not raise on missing data; unresolved citations have `bibliography_ref = None` and a lower confidence.
        """
        # TODO: Implement in-text citation extraction
        # This should scan paper text for patterns like [1], [2], (Author, Year)
        # and map them to the provided bibliography_entries
        logger.warning(
            "extract_citations is not yet implemented for paper %s; returning empty list",
            paper_id
        )
        return []

# Factory functions for easy setup
def create_arxiv_citation_toolkit(arango_client) -> tuple[UniversalBibliographyExtractor, ArangoCitationStorage]:
    """
    Create a toolkit configured to extract bibliographies from ArXiv papers and store results in ArangoDB.

    Returns:
        Tuple[UniversalBibliographyExtractor, ArangoCitationStorage]: A tuple containing a bibliography extractor wired to an ArangoDocumentProvider and an Arango-backed citation storage instance.
    """
    document_provider = ArangoDocumentProvider(arango_client)
    storage = ArangoCitationStorage(arango_client)
    extractor = UniversalBibliographyExtractor(document_provider)
    return extractor, storage

def create_filesystem_citation_toolkit(file_path: str, output_path: str) -> tuple[UniversalBibliographyExtractor, JSONCitationStorage]:
    """
    Create a filesystem-backed citation toolkit that reads papers from plain-text files and writes results as JSON.

    Parameters:
        file_path (str): Directory containing paper text files (each expected as "<paper_id>.txt").
        output_path (str): Directory where JSON output files (bibliography.json, citations.json) will be written; created if missing.

    Returns:
        Tuple[UniversalBibliographyExtractor, JSONCitationStorage]: A bibliography extractor configured with a FileSystemDocumentProvider and a JSONCitationStorage for persisting results.
    """
    document_provider = FileSystemDocumentProvider(file_path)
    storage = JSONCitationStorage(output_path)
    extractor = UniversalBibliographyExtractor(document_provider)
    return extractor, storage

# Main function to test the toolkit
def main():
    """
    Run a simple CLI test/demo of the Universal Academic Citation Toolkit.

    This convenience entrypoint exercises the extractor and storage components against a small
    set of example paper IDs. It:
    - Creates an Arango-backed toolkit using create_arxiv_citation_toolkit.
    - For each test paper, attempts to extract the bibliography section, parse bibliography
      entries, prints a short human-readable summary to stdout, and persists parsed entries
      via the configured storage backend.

    Notes:
    - Intended for local/manual testing and demonstration only (prints to stdout and performs
      network and file I/O).
    - Expects an ArangoDB instance reachable at the configured host in the code; ARANGO_PASSWORD
      may be read from the environment in some setups.
    - Side effects: network access to ArangoDB, console output, and storage writes via the
      chosen CitationStorage implementation.
    """
    from arango import ArangoClient

    # Read connection config from environment
    arango_host = os.environ.get('ARANGO_HOST', 'http://localhost:8529')
    arango_password = os.environ.get('ARANGO_PASSWORD')
    if not arango_password:
        raise ValueError("ARANGO_PASSWORD environment variable required")

    # Create client using environment-configured host
    client = ArangoClient(hosts=arango_host)

    extractor, storage = create_arxiv_citation_toolkit(client)

    print("🌍 Universal Academic Citation Toolkit")
    print("=" * 50)
    print("SOURCE-AGNOSTIC: Works with ArXiv, SSRN, PubMed, Law Libraries, etc.")
    print()

    # Test on our papers
    test_papers = ['1301_3781']  # Start with one

    for paper_id in test_papers:
        print(f"📄 Processing paper: {paper_id}")

        # Extract bibliography
        bibliography_text = extractor.extract_bibliography_section(paper_id)
        if bibliography_text:
            print(f"  ✅ Found bibliography ({len(bibliography_text)} chars)")

            entries = extractor.parse_bibliography_entries(bibliography_text, paper_id)
            print(f"  📚 Parsed {len(entries)} bibliography entries")

            # Show sample entries
            for i, entry in enumerate(entries[:3], 1):
                print(f"    {i}. [{entry.entry_number}] {entry.title or 'No title'}")
                if entry.arxiv_id:
                    print(f"       ArXiv: {entry.arxiv_id}")
                if entry.authors:
                    print(f"       Authors: {', '.join(entry.authors[:2])}")
                print(f"       Confidence: {entry.confidence:.2f}")

            if len(entries) > 3:
                print(f"    ... and {len(entries) - 3} more entries")

            # Store results
            if storage.store_bibliography_entries(entries):
                print("  💾 Stored bibliography entries")

        else:
            print("  ❌ No bibliography found")

if __name__ == "__main__":
    main()
