"""Unit tests for document parsers, scanned PDF detection, and recursive chunking (§4.2, §7.7)."""

from __future__ import annotations

import io
from pathlib import Path
import docx
from pypdf import PdfWriter
import pytest

from app.core.errors import ValidationError
from app.core.ingestion import (
    SCANNED_PDF_WARNING,
    chunk_segments,
    parse_document,
    recursive_split_text,
)
from samples.generate_samples import build_multipage_pdf


def test_recursive_chunker_prefers_paragraph_and_sentence_boundaries() -> None:
    para1 = "The Astra NX-200 operates from 12 V to 48 V DC. Nominal power draw is 18 W."
    para2 = "Thermal pads must be replaced after 24 months. Torque M3 screws to 0.65 N-m."
    para3 = "Hold RECOV for 7 seconds during boot. Connect to 192.168.200.1 on port 8443."
    full_text = f"{para1}\n\n{para2}\n\n{para3}"

    chunks = recursive_split_text(full_text, chunk_size=110, chunk_overlap=25)
    assert len(chunks) >= 3
    for chunk in chunks:
        assert len(chunk) <= 110
    assert "12 V to 48 V DC" in chunks[0]
    assert "192.168.200.1" in chunks[-1]


def test_recursive_chunker_handles_unbroken_long_string_and_overlap() -> None:
    unbroken = "A" * 250
    chunks = recursive_split_text(unbroken, chunk_size=100, chunk_overlap=20)
    assert len(chunks) >= 3
    assert all(len(c) <= 100 for c in chunks)


def test_pdf_parser_preserves_page_numbers(tmp_path: Path) -> None:
    pdf_file = tmp_path / "spec.pdf"
    build_multipage_pdf(
        [
            ["Page One Heading", "The ARM Cortex-A78AE processor runs at 2.2 GHz with 8 GB ECC RAM."],
            ["Page Two Heading", "Dual redundant DC inputs accept 12 V to 48 V DC continuous power."],
        ],
        pdf_file,
    )

    parsed = parse_document(pdf_file.read_bytes(), "spec.pdf")
    assert not parsed.is_scanned_pdf
    assert parsed.warning is None
    assert len(parsed.segments) == 2
    assert parsed.segments[0].page == 1
    assert "Cortex-A78AE" in parsed.segments[0].text
    assert parsed.segments[1].page == 2
    assert "48 V DC" in parsed.segments[1].text

    chunks = chunk_segments(parsed.segments, file_hash="abc123", chunk_size=500, chunk_overlap=50)
    assert len(chunks) == 2
    assert chunks[0].location_label == "Page 1"
    assert chunks[1].location_label == "Page 2"


def test_blank_or_scanned_pdf_is_flagged_not_silently_indexed() -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.add_blank_page(width=612, height=792)
    buf = io.BytesIO()
    writer.write(buf)

    parsed = parse_document(buf.getvalue(), "scanned_drawing.pdf")
    assert parsed.is_scanned_pdf is True
    assert parsed.warning == SCANNED_PDF_WARNING
    assert parsed.segments == []


def test_docx_parser_keeps_heading_as_section_label() -> None:
    doc = docx.Document()
    doc.add_heading("Safety Precautions", level=1)
    doc.add_paragraph("Disconnect 24 V DC supply before opening the chassis.")
    doc.add_heading("Torque Specifications", level=2)
    doc.add_paragraph("Tighten M3 lid screws to 0.65 N-m.")
    buf = io.BytesIO()
    doc.save(buf)

    parsed = parse_document(buf.getvalue(), "sop.docx")
    assert len(parsed.segments) == 2
    assert parsed.segments[0].section == "Safety Precautions"
    assert "Disconnect 24 V DC" in parsed.segments[0].text
    assert parsed.segments[1].section == "Torque Specifications"
    assert "0.65 N-m" in parsed.segments[1].text


def test_unsupported_file_extension_rejected() -> None:
    with pytest.raises(ValidationError, match="Unsupported file format"):
        parse_document(b"fake spreadsheet", "budget.xlsx")
