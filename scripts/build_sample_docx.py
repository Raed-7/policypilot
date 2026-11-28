"""Build the Word sample policy in data/sample_policies."""

import argparse
from pathlib import Path

from docx import Document
from docx.shared import Pt

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT.parent / "data" / "sample_policies"

CONTENT = [
    ("title", "Health and Safety Policy"),
    ("p", "Version 1.3 - effective 1 March 2026. Owner: Facilities."),
    ("h1", "Purpose"),
    ("p", "Halden Analytics is committed to providing a safe and healthy workplace for employees, contractors and visitors. "
          "Everyone has a duty to take reasonable care of their own safety and the safety of others."),
    ("h1", "Reporting accidents"),
    ("p", "All accidents, injuries and near misses must be reported in the SafeLog app within 24 hours, even if no one was hurt. "
          "Facilities reviews every report and shares lessons learned with the relevant team."),
    ("h1", "Fire safety"),
    ("p", "Fire alarms are tested every Wednesday at 11:00. If the alarm sounds at any other time, stop work, leave by the nearest "
          "exit and go straight to your office's assembly point. Do not use the lifts or stop to collect belongings."),
    ("h2", "Assembly points"),
    ("li", "Leeds: the car park on Wellington Street."),
    ("li", "Manchester: the north entrance of Piccadilly Gardens."),
    ("h1", "First aid"),
    ("p", "Every office has trained first aiders and a stocked first aid kit. Their names are displayed on the noticeboard by reception."),
    ("table", [["Office", "Trained first aiders", "First aid kit location"],
               ["Leeds", "4", "Kitchen on floor 2"],
               ["Manchester", "2", "Behind the reception desk"]]),
    ("h1", "Display screen equipment"),
    ("p", "Employees who use a screen for more than one hour a day are entitled to a free eye test every two years. "
          "If the optician confirms you need glasses specifically for screen work, Halden contributes up to £60 towards them."),
    ("h1", "Lone working"),
    ("p", "Anyone working alone in an office outside 07:00 to 19:00 must check in with building security through the LoneSafe app "
          "every two hours. Hazardous tasks, such as using ladders, must never be carried out while working alone."),
]


def build(out_dir: Path) -> Path:
    doc = Document()
    doc.core_properties.title = "Health and Safety Policy"
    doc.core_properties.author = "Halden Analytics Ltd"
    doc.styles["Normal"].font.size = Pt(11)
    for kind, value in CONTENT:
        if kind == "title":
            doc.add_heading(value, level=0)
        elif kind == "h1":
            doc.add_heading(value, level=1)
        elif kind == "h2":
            doc.add_heading(value, level=2)
        elif kind == "li":
            doc.add_paragraph(value, style="List Bullet")
        elif kind == "table":
            table = doc.add_table(rows=0, cols=len(value[0]))
            table.style = "Light Grid Accent 1"
            for row in value:
                cells = table.add_row().cells
                for cell, text in zip(cells, row):
                    cell.text = text
        else:
            doc.add_paragraph(value)
    out = out_dir / "health-and-safety-policy.docx"
    doc.save(out)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(f"wrote {build(args.out)}")


if __name__ == "__main__":
    main()
