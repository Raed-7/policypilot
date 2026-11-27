import re
import unicodedata

_LIGATURES = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl"}
_QUOTES = {"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", " ": " "}
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_SOFT_WRAP = re.compile(r"(?<![.!?\n])\n(?!\n|\s*[-*•]\s|\s*\d+[.)]\s)")
_PAGE_FOOTER = re.compile(r"^\s*(page\s+\d+(\s+of\s+\d+)?|\d+\s*/\s*\d+)\s*$", re.IGNORECASE | re.MULTILINE)
_MULTI_SPACE = re.compile(r"[ \t]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Normalise extracted text without changing its meaning."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    for src, dst in {**_LIGATURES, **_QUOTES}.items():
        text = text.replace(src, dst)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or not unicodedata.category(ch).startswith("C"))
    text = _PAGE_FOOTER.sub("", text)
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = _SOFT_WRAP.sub(" ", text)
    text = _MULTI_SPACE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()
