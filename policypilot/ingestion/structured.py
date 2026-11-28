"""Loaders for formats with explicit heading structure: Word documents and HTML pages."""

import io
import re
from html.parser import HTMLParser

from policypilot.ingestion.cleaning import clean_text
from policypilot.models import Section

Block = tuple[int, str]  # (heading level, text); level 0 means body text

_HEADING_STYLE = re.compile(r"^(?:heading|überschrift|titre)\s*(\d)$", re.IGNORECASE)


def sections_from_blocks(blocks: list[Block]) -> tuple[list[Section], str | None]:
    """Group body text under the nearest heading path, e.g. "Leave > Sick leave"."""
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    buffer: list[str] = []
    title = None

    def flush():
        body = clean_text("\n\n".join(buffer))
        if body:
            path = [h for _, h in stack[1:]] if len(stack) > 1 else [h for _, h in stack]
            sections.append(Section(text=body, heading=" > ".join(path)))
        buffer.clear()

    for level, text in blocks:
        text = text.strip()
        if not text:
            continue
        if level:
            flush()
            if title is None and level == 1:
                title = text
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, text))
        else:
            buffer.append(text)
    flush()
    return sections, title


def table_to_sentences(rows: list[list[str]]) -> list[str]:
    """Render each table row as a self-describing line, e.g. "Office: Leeds; First aid kit location: Kitchen.".

    Keeping the header with every value lets a single row be retrieved and quoted on its own.
    """
    rows = [r for r in rows if any(r)]
    if not rows:
        return []
    if len(rows) == 1:
        return [" | ".join(c for c in rows[0] if c) + "."]
    header, *body = rows
    lines = []
    for row in body:
        pairs = [f"{h}: {v}" if h else v for h, v in zip(header, row) if v]
        if pairs:
            lines.append("; ".join(pairs) + ".")
    return lines


def _docx_level(paragraph) -> int:
    style = paragraph.style.name if paragraph.style is not None else ""
    if style.lower() == "title":
        return 1
    match = _HEADING_STYLE.match(style)
    return int(match.group(1)) + 1 if match else 0


def load_docx(data: bytes) -> tuple[list[Section], str | None]:
    from docx import Document as DocxDocument

    doc = DocxDocument(io.BytesIO(data))
    blocks: list[Block] = []
    body = doc.element.body
    for child in body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            from docx.text.paragraph import Paragraph

            para = Paragraph(child, doc)
            text = para.text
            if para.style is not None and para.style.name.lower().startswith("list"):
                text = f"- {text}"
            blocks.append((_docx_level(para), text))
        elif tag == "tbl":
            from docx.table import Table

            rows = [[c.text.strip() for c in row.cells] for row in Table(child, doc).rows]
            blocks.extend((0, line) for line in table_to_sentences(rows))
    sections, title = sections_from_blocks(blocks)
    core_title = (doc.core_properties.title or "").strip()
    return sections, core_title or title


class _HTMLBlocks(HTMLParser):
    SKIP = {"script", "style", "nav", "header", "footer", "noscript", "svg", "form", "aside", "button"}
    BLOCK = {"p", "div", "li", "section", "article", "blockquote", "dd", "dt", "br", "ul", "ol", "main"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.blocks: list[Block] = []
        self.title = ""
        self._buf: list[str] = []
        self._level = 0
        self._skip = 0
        self._in_title = False
        self._table: list[list[str]] | None = None
        self._cell: list[str] | None = None

    def _flush(self):
        text = re.sub(r"\s+", " ", "".join(self._buf)).strip()
        if text:
            self.blocks.append((self._level, text))
        self._buf.clear()

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1
        elif self._skip:
            return
        elif tag == "title":
            self._in_title = True
        elif tag == "table":
            self._flush()
            self._table = []
        elif tag == "tr" and self._table is not None:
            self._table.append([])
        elif tag in {"td", "th"} and self._table is not None:
            self._cell = []
        elif re.fullmatch(r"h[1-6]", tag):
            self._flush()
            self._level = int(tag[1])
        elif tag in self.BLOCK:
            self._flush()
            if tag == "li":
                self._buf.append("- ")

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self._skip = max(0, self._skip - 1)
        elif self._skip:
            return
        elif tag == "title":
            self._in_title = False
        elif tag in {"td", "th"} and self._cell is not None:
            if self._table:
                self._table[-1].append(re.sub(r"\s+", " ", "".join(self._cell)).strip())
            self._cell = None
        elif tag == "table" and self._table is not None:
            self.blocks.extend((0, line) for line in table_to_sentences(self._table))
            self._table = None
        elif re.fullmatch(r"h[1-6]", tag):
            self._flush()
            self._level = 0
        elif tag in self.BLOCK:
            self._flush()

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif self._skip:
            return
        elif self._cell is not None:
            self._cell.append(data)
        elif self._table is None:
            self._buf.append(data)


def load_html(data: bytes) -> tuple[list[Section], str | None]:
    parser = _HTMLBlocks()
    parser.feed(data.decode("utf-8", errors="replace"))
    parser.close()
    parser._flush()
    sections, heading_title = sections_from_blocks(parser.blocks)
    return sections, parser.title.strip() or heading_title
