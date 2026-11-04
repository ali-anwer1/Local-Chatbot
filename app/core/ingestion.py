"""Document parsers (PDF, DOCX, TXT, MD) and recursive character chunker (§4.2)."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
from pathlib import Path
import re

import docx
from docx.document import Document as DocxDocument
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph
from pypdf import PdfReader

from app.core.errors import ValidationError


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
SCANNED_PDF_WARNING = "likely scanned, not supported"
MIN_NONWS_CHARS_PER_PDF_PAGE = 20

DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]


@dataclass(frozen=True)
class ParsedSegment:
    """Logical text unit extracted from a source file with page or section provenance."""

    text: str
    source_file: str
    page: int | None = None
    section: str | None = None


@dataclass(frozen=True)
class ParseResult:
    """Result of parsing an uploaded document."""

    segments: list[ParsedSegment]
    is_scanned_pdf: bool = False
    warning: str | None = None


def format_location_label(page: int | None, section: str | None) -> str | None:
    """Format 1-based PDF page number or document section heading for citations."""
    if page is not None:
        return f"Page {page}"
    if section:
        return f"Section: {section}"
    return None


@dataclass(frozen=True)
class Chunk:
    """An indexable document chunk with full citation provenance."""

    text: str
    source_file: str
    chunk_index: int
    file_hash: str
    page: int | None = None
    section: str | None = None

    @property
    def location_label(self) -> str | None:
        return format_location_label(self.page, self.section)


def compute_sha256(data: bytes) -> str:
    """Compute hex SHA-256 digest for file deduplication."""
    return hashlib.sha256(data).hexdigest()


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse 3+ consecutive blank lines down to paragraph break
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _decode_text_bytes(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def parse_pdf(data: bytes, filename: str) -> ParseResult:
    """Parse PDF via pypdf while preserving 1-based page numbers and detecting scanned PDFs."""
    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as exc:
        raise ValidationError(
            f"Could not read PDF file '{filename}': file appears corrupted or encrypted.",
            details=str(exc),
        ) from exc

    num_pages = len(reader.pages)
    if num_pages == 0:
        return ParseResult(segments=[], is_scanned_pdf=True, warning=SCANNED_PDF_WARNING)

    segments: list[ParsedSegment] = []
    total_nonws_chars = 0

    for page_idx, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        cleaned = _normalize_whitespace(raw_text)
        nonws = len(re.sub(r"\s+", "", cleaned))
        total_nonws_chars += nonws
        if cleaned:
            segments.append(
                ParsedSegment(
                    text=cleaned,
                    source_file=filename,
                    page=page_idx,
                )
            )

    avg_nonws = total_nonws_chars / max(1, num_pages)
    if total_nonws_chars == 0 or avg_nonws < MIN_NONWS_CHARS_PER_PDF_PAGE:
        return ParseResult(
            segments=[],
            is_scanned_pdf=True,
            warning=SCANNED_PDF_WARNING,
        )

    return ParseResult(segments=segments)


def _iter_docx_blocks(doc: DocxDocument):
    """Yield Paragraph and Table blocks in document order."""
    if hasattr(doc, "iter_inner_content"):
        yield from doc.iter_inner_content()
        return

    body = doc.element.body
    for child in body.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, doc)
        elif isinstance(child, CT_Tbl):
            yield Table(child, doc)


def parse_docx(data: bytes, filename: str) -> ParseResult:
    """Parse DOCX via python-docx, preserving headings as section labels."""
    try:
        doc = docx.Document(io.BytesIO(data))
    except Exception as exc:
        raise ValidationError(
            f"Could not read DOCX file '{filename}': invalid Word document.",
            details=str(exc),
        ) from exc

    segments: list[ParsedSegment] = []
    current_section: str = "General"
    buffer: list[str] = []

    def flush_section() -> None:
        if not buffer:
            return
        combined = _normalize_whitespace("\n\n".join(buffer))
        buffer.clear()
        if combined:
            segments.append(
                ParsedSegment(
                    text=combined,
                    source_file=filename,
                    section=current_section,
                )
            )

    for block in _iter_docx_blocks(doc):
        if isinstance(block, Paragraph):
            text = block.text.strip()
            if not text:
                continue
            style_name = (block.style.name if block.style and block.style.name else "").lower()
            if style_name.startswith("heading") or style_name == "title":
                flush_section()
                current_section = text
            buffer.append(text)
        elif isinstance(block, Table):
            table_lines: list[str] = []
            for row in block.rows:
                cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
                if any(cells):
                    table_lines.append(" | ".join(cells))
            if table_lines:
                buffer.append("\n".join(table_lines))

    flush_section()
    return ParseResult(segments=segments)


def parse_markdown(data: bytes, filename: str) -> ParseResult:
    """Parse Markdown file, grouping content under ATX headings (# .. ####) as section labels."""
    raw_text = _decode_text_bytes(data)
    lines = raw_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    segments: list[ParsedSegment] = []
    current_section: str | None = None
    buffer: list[str] = []
    heading_pattern = re.compile(r"^(#{1,4})\s+(.+?)\s*$")

    def flush() -> None:
        if not buffer:
            return
        combined = _normalize_whitespace("\n".join(buffer))
        buffer.clear()
        if combined:
            segments.append(
                ParsedSegment(
                    text=combined,
                    source_file=filename,
                    section=current_section,
                )
            )

    for line in lines:
        match = heading_pattern.match(line)
        if match:
            flush()
            current_section = match.group(2).strip()
        buffer.append(line)

    flush()
    return ParseResult(segments=segments)


def parse_txt(data: bytes, filename: str) -> ParseResult:
    """Parse plain text file."""
    cleaned = _normalize_whitespace(_decode_text_bytes(data))
    if not cleaned:
        return ParseResult(segments=[])
    return ParseResult(segments=[ParsedSegment(text=cleaned, source_file=filename)])


def parse_document(data: bytes, filename: str) -> ParseResult:
    """Dispatch document bytes to the appropriate format parser."""
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValidationError(
            f"Unsupported file format '{ext or 'none'}'. Supported formats: {allowed}."
        )
    if ext == ".pdf":
        return parse_pdf(data, filename)
    if ext == ".docx":
        return parse_docx(data, filename)
    if ext == ".md":
        return parse_markdown(data, filename)
    return parse_txt(data, filename)


def _split_into_atomic_pieces(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    """Recursively break text into pieces no larger than chunk_size."""
    if len(text) <= chunk_size:
        return [text] if text else []

    if not separators or separators[0] == "":
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    sep = separators[0]
    next_seps = separators[1:]

    if sep not in text:
        return _split_into_atomic_pieces(text, chunk_size, next_seps)

    raw_parts = text.split(sep)
    pieces: list[str] = []
    for idx, part in enumerate(raw_parts):
        # Re-attach separator to preceding piece so punctuation & spacing stay intact
        piece = part + (sep if idx < len(raw_parts) - 1 else "")
        if not piece:
            continue
        if len(piece) <= chunk_size:
            pieces.append(piece)
        else:
            pieces.extend(_split_into_atomic_pieces(piece, chunk_size, next_seps))
    return pieces


def recursive_split_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    separators: list[str] | None = None,
) -> list[str]:
    """Split text recursively preferring paragraph and sentence boundaries with overlap."""
    cleaned = text.strip()
    if not cleaned:
        return []
    if chunk_size <= 0:
        raise ValidationError("chunk_size must be positive.")

    effective_overlap = max(0, min(chunk_overlap, chunk_size - 1))
    seps = separators if separators is not None else DEFAULT_SEPARATORS
    atomic_pieces = _split_into_atomic_pieces(cleaned, chunk_size, seps)

    chunks: list[str] = []
    current_pieces: list[str] = []
    current_len = 0

    for piece in atomic_pieces:
        piece_len = len(piece)
        if current_len + piece_len > chunk_size and current_pieces:
            chunk_str = "".join(current_pieces).strip()
            if chunk_str:
                chunks.append(chunk_str)

            # Retain trailing pieces up to effective_overlap characters
            if effective_overlap > 0:
                overlap_pieces: list[str] = []
                overlap_len = 0
                for prev in reversed(current_pieces):
                    if overlap_len + len(prev) > effective_overlap and overlap_pieces:
                        break
                    if len(prev) > effective_overlap and not overlap_pieces:
                        overlap_pieces.insert(0, prev[-effective_overlap:])
                        overlap_len = effective_overlap
                        break
                    overlap_pieces.insert(0, prev)
                    overlap_len += len(prev)
                # Ensure adding the new piece won't still exceed chunk_size
                while overlap_pieces and (overlap_len + piece_len > chunk_size):
                    removed = overlap_pieces.pop(0)
                    overlap_len -= len(removed)
                current_pieces = overlap_pieces
                current_len = overlap_len
            else:
                current_pieces = []
                current_len = 0

        current_pieces.append(piece)
        current_len += piece_len

    if current_pieces:
        final_str = "".join(current_pieces).strip()
        if final_str and (not chunks or final_str != chunks[-1]):
            chunks.append(final_str)

    return chunks


def chunk_segments(
    segments: list[ParsedSegment],
    *,
    file_hash: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[Chunk]:
    """Chunk parsed document segments while preserving page/section provenance."""
    all_chunks: list[Chunk] = []
    chunk_idx = 0

    for segment in segments:
        split_texts = recursive_split_text(
            segment.text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        for text in split_texts:
            all_chunks.append(
                Chunk(
                    text=text,
                    source_file=segment.source_file,
                    chunk_index=chunk_idx,
                    file_hash=file_hash,
                    page=segment.page,
                    section=segment.section,
                )
            )
            chunk_idx += 1

    return all_chunks
