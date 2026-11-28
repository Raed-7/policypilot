import io

from docx import Document as DocxDocument

from policypilot.ingestion.loaders import load_bytes
from policypilot.ingestion.structured import sections_from_blocks, table_to_sentences
from tests.conftest import SAMPLE_DOCS


def _docx_bytes() -> bytes:
    doc = DocxDocument()
    doc.add_heading("Leave Policy", level=0)
    doc.add_paragraph("Version 2.")
    doc.add_heading("Sick leave", level=1)
    doc.add_paragraph("Call your manager by 09:30.")
    doc.add_heading("Fit notes", level=2)
    doc.add_paragraph("Needed from day eight.", style="List Bullet")
    table = doc.add_table(rows=2, cols=2)
    table.rows[0].cells[0].text, table.rows[0].cells[1].text = "Service", "Full pay"
    table.rows[1].cells[0].text, table.rows[1].cells[1].text = "Year one", "10 days"
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def test_docx_headings_lists_and_tables():
    doc = load_bytes(_docx_bytes(), "leave.docx")
    assert doc.file_type == "docx"
    assert doc.title == "Leave Policy"
    headings = [s.heading for s in doc.sections]
    assert headings == ["Leave Policy", "Sick leave", "Sick leave > Fit notes"]
    fit_notes = doc.sections[-1].text
    assert "- Needed from day eight." in fit_notes
    assert "Service: Year one; Full pay: 10 days." in fit_notes


def test_sample_word_policy_loads():
    doc = load_bytes((SAMPLE_DOCS / "health-and-safety-policy.docx").read_bytes(), "health-and-safety-policy.docx")
    assert doc.title == "Health and Safety Policy"
    first_aid = next(s for s in doc.sections if s.heading == "First aid")
    assert "First aid kit location: Kitchen on floor 2" in first_aid.text


HTML = b"""<html><head><title>Travel Guide</title><style>.x{color:red}</style></head>
<body><nav>Home | About</nav><h1>Travel Guide</h1><p>Intro &amp; overview.</p>
<h2>Rail</h2><p>Book   standard
class.</p><ul><li>Use TripDesk</li><li>Keep tickets</li></ul>
<h2>Hotels</h2><table><tr><th>City</th><th>Limit</th></tr><tr><td>London</td><td>&pound;150</td></tr></table>
<script>alert("x")</script><footer>Copyright 2026</footer></body></html>"""


def test_html_structure_and_boilerplate_removal():
    doc = load_bytes(HTML, "guide.html")
    assert doc.file_type == "html"
    assert doc.title == "Travel Guide"
    by_heading = {s.heading: s.text for s in doc.sections}
    assert by_heading["Rail"] == "Book standard class.\n\n- Use TripDesk\n\n- Keep tickets"
    assert by_heading["Hotels"] == "City: London; Limit: £150."
    full = " ".join(by_heading.values())
    assert "alert" not in full and "Copyright" not in full and "About" not in full


def test_table_to_sentences():
    assert table_to_sentences([["City", "Limit"], ["Leeds", "£110"], ["", ""]]) == ["City: Leeds; Limit: £110."]
    assert table_to_sentences([["only", "row"]]) == ["only | row."]
    assert table_to_sentences([]) == []


def test_sections_from_blocks_nests_headings():
    sections, title = sections_from_blocks([(1, "Doc"), (0, "intro"), (2, "A"), (3, "B"), (0, "deep"), (2, "C"), (0, "flat")])
    assert title == "Doc"
    assert [(s.heading, s.text) for s in sections] == [("Doc", "intro"), ("A > B", "deep"), ("C", "flat")]
