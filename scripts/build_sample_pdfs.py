"""Render the PDF sample policies from their text sources in scripts/pdf_sources."""

import argparse
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT.parent / "data" / "sample_policies"


def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("t", parent=base["Title"], fontSize=18, spaceAfter=4),
        "meta": ParagraphStyle("m", parent=base["Normal"], fontSize=8.5, textColor="#555555", spaceAfter=10),
        "h2": ParagraphStyle("h", parent=base["Heading2"], fontSize=13, spaceBefore=10, spaceAfter=4),
        "body": ParagraphStyle("b", parent=base["BodyText"], fontSize=10.5, leading=14.5, spaceAfter=6),
    }


def render(source: Path, out_dir: Path) -> Path:
    styles = _styles()
    lines = source.read_text(encoding="utf-8").splitlines()
    title = lines[0].lstrip("# ").strip()
    story = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        if line == "<<PAGE>>":
            story.append(PageBreak())
        elif i == 0:
            story.append(Paragraph(title, styles["title"]))
        elif line.startswith("## "):
            story.append(Paragraph(line[3:], styles["h2"]))
        elif i == 1:
            story.append(Paragraph(line, styles["meta"]))
        else:
            story.append(Paragraph(line.replace("&", "&amp;"), styles["body"]))
    story.append(Spacer(1, 4 * mm))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor("#777777")
        canvas.drawString(20 * mm, 12 * mm, f"Halden Analytics Ltd - {title}")
        canvas.drawRightString(190 * mm, 12 * mm, f"Page {doc.page}")
        canvas.restoreState()

    out = out_dir / f"{source.stem}.pdf"
    doc = SimpleDocTemplate(
        str(out), pagesize=A4, title=title, author="Halden Analytics Ltd",
        leftMargin=20 * mm, rightMargin=20 * mm, topMargin=20 * mm, bottomMargin=22 * mm,
        invariant=1,
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for source in sorted((ROOT / "pdf_sources").glob("*.txt")):
        print(f"wrote {render(source, args.out)}")


if __name__ == "__main__":
    main()
