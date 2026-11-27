import hashlib
import io
import re
from pathlib import Path

from pypdf import PdfReader

from policypilot.ingestion.cleaning import clean_text
from policypilot.models import Document, Section

SUPPORTED_EXTENSIONS = {".pdf", ".md", ".markdown", ".txt"}
_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_TXT_HEADING = re.compile(r"^(?:\d+(?:\.\d+)*[.)]?\s+)?[A-Z][A-Za-z0-9 ,&'()/-]{2,80}$")


class UnsupportedFileType(ValueError):
    pass


def _doc_id(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def _title_from_filename(name: str) -> str:
    stem = Path(name).stem
    return re.sub(r"[-_]+", " ", stem).strip().title()


def _load_pdf(data: bytes) -> tuple[list[Section], str | None]:
    reader = PdfReader(io.BytesIO(data))
    sections = []
    for number, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if text:
            sections.append(Section(text=text, page=number))
    title = None
    if reader.metadata and reader.metadata.title:
        title = str(reader.metadata.title).strip() or None
    return sections, title


def _split_markdown(text: str) -> tuple[list[Section], str | None]:
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    buffer: list[str] = []
    title = None

    def flush():
        body = clean_text("\n".join(buffer))
        if body:
            heading = " > ".join(h for _, h in stack[1:]) if len(stack) > 1 else (stack[0][1] if stack else "")
            sections.append(Section(text=body, heading=heading))
        buffer.clear()

    in_code = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
        match = None if in_code else _MD_HEADING.match(line)
        if match:
            flush()
            level, heading = len(match.group(1)), match.group(2).strip()
            if title is None and level == 1:
                title = heading
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading))
        else:
            buffer.append(line)
    flush()
    return sections, title


def _split_plain_text(text: str) -> tuple[list[Section], str | None]:
    """Split plain text on heading-like lines (short, capitalised, surrounded by blank lines)."""
    lines = text.replace("\r\n", "\n").split("\n")
    sections: list[Section] = []
    heading, buffer = "", []
    title = None

    def flush():
        body = clean_text("\n".join(buffer))
        if body:
            sections.append(Section(text=body, heading=heading))
        buffer.clear()

    for i, raw in enumerate(lines):
        line = raw.strip()
        prev_blank = i == 0 or not lines[i - 1].strip()
        next_blank = i + 1 >= len(lines) or not lines[i + 1].strip()
        is_heading = (
            line
            and prev_blank
            and next_blank
            and len(line.split()) <= 10
            and not line.endswith((".", ",", ";"))
            and (line.isupper() or _TXT_HEADING.match(line))
        )
        if is_heading:
            if title is None and not sections and not any(b.strip() for b in buffer):
                title = line.title() if line.isupper() else line
            flush()
            heading = line.title() if line.isupper() else line
        else:
            buffer.append(raw)
    flush()
    return sections, title


def load_bytes(data: bytes, filename: str) -> Document:
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileType(f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")

    if ext == ".pdf":
        sections, title = _load_pdf(data)
        file_type = "pdf"
    else:
        text = data.decode("utf-8", errors="replace")
        if ext in {".md", ".markdown"}:
            sections, title = _split_markdown(text)
            file_type = "markdown"
        else:
            sections, title = _split_plain_text(text)
            file_type = "text"

    name = Path(filename).name
    return Document(
        doc_id=_doc_id(data),
        source=name,
        title=title or _title_from_filename(name),
        sections=sections,
        file_type=file_type,
        metadata={"size_bytes": len(data)},
    )


def load_document(path: str | Path) -> Document:
    path = Path(path)
    return load_bytes(path.read_bytes(), path.name)


def load_directory(directory: str | Path) -> list[Document]:
    directory = Path(directory)
    return [
        load_document(p)
        for p in sorted(directory.iterdir())
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
