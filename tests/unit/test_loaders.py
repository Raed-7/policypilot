import pytest

from policypilot.ingestion import UnsupportedFileType, load_directory
from policypilot.ingestion.loaders import load_bytes
from tests.conftest import SAMPLE_DOCS

MARKDOWN = b"""# Travel Policy

Intro text.

## Flights

Economy for short flights.

### Long haul

Premium economy over six hours.
"""


def test_markdown_sections_keep_heading_path():
    doc = load_bytes(MARKDOWN, "travel.md")
    assert doc.title == "Travel Policy"
    assert [s.heading for s in doc.sections] == ["Travel Policy", "Flights", "Flights > Long haul"]
    assert doc.sections[2].text == "Premium economy over six hours."
    assert all(s.page is None for s in doc.sections)


def test_plain_text_headings_are_detected():
    text = b"LEAVE POLICY\n\nIntro line.\n\nSICK LEAVE\n\nCall your manager by 09:30.\n"
    doc = load_bytes(text, "leave.txt")
    assert doc.title == "Leave Policy"
    assert doc.sections[-1].heading == "Sick Leave"
    assert "09:30" in doc.sections[-1].text


def test_pdf_pages_and_headings_are_preserved():
    doc = load_bytes((SAMPLE_DOCS / "expenses-and-travel-policy.pdf").read_bytes(), "expenses-and-travel-policy.pdf")
    assert doc.file_type == "pdf"
    assert doc.title == "Expenses and Business Travel Policy"
    assert {s.page for s in doc.sections} == {1, 2, 3}
    mileage = next(s for s in doc.sections if s.heading == "Mileage and taxis")
    assert mileage.page == 2
    assert "45p per mile" in mileage.text


def test_pdf_running_footer_is_removed():
    doc = load_bytes((SAMPLE_DOCS / "employee-handbook.pdf").read_bytes(), "employee-handbook.pdf")
    assert not any("Page 2" in s.text or "Halden Analytics Ltd -" in s.text for s in doc.sections)


def test_doc_id_is_content_hash():
    assert load_bytes(b"same", "a.txt").doc_id == load_bytes(b"same", "b.txt").doc_id
    assert load_bytes(b"same", "a.txt").doc_id != load_bytes(b"different", "a.txt").doc_id


def test_rejects_unsupported_extension():
    with pytest.raises(UnsupportedFileType):
        load_bytes(b"data", "sheet.xlsx")


def test_load_directory_reads_all_supported_files():
    docs = load_directory(SAMPLE_DOCS)
    assert len(docs) == 7
    assert {d.file_type for d in docs} == {"pdf", "markdown", "text", "docx"}
