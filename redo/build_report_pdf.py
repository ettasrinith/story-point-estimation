"""Render REPORT.md into REPORT.pdf using headless Chrome.

The repository's report is written in Markdown. This script converts the small subset of Markdown
the report actually uses (headings, paragraphs, bold/italic, inline code, links, lists, tables,
block quotes and fenced code blocks) into a print-styled HTML page, then asks Chrome to print it
to PDF. No third-party Python packages are needed.

    python redo/build_report_pdf.py

It is also called automatically by ``redo/finalize_main.py`` so the PDF always matches the
latest generated results tables.
"""
from __future__ import annotations

import html
import re
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "REPORT.md"
DST = ROOT / "REPORT.pdf"

CHROME_CANDIDATES = [
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
]


def inline(text: str) -> str:
    """Convert inline Markdown to HTML, preserving the entities used in generated tables."""
    text = text.replace("&mdash;", "—").replace("&ndash;", "–")
    codes: list[str] = []

    def stash(m):
        codes.append(m.group(1))
        return f"\x00{len(codes) - 1}\x00"

    text = re.sub(r"`([^`]+)`", stash, text)
    text = html.escape(text, quote=False)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)
    text = re.sub(r'(?<!["=])(https?://[^\s<),]+)', r'<a href="\1">\1</a>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", text)
    text = re.sub(r"\x00(\d+)\x00",
                  lambda m: f"<code>{html.escape(codes[int(m.group(1))])}</code>", text)
    return text


def is_separator(line: str) -> bool:
    return bool(re.match(r"^\s*\|[\s:|-]+\|\s*$", line)) and "-" in line


def cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def md_to_html(md: str) -> str:
    lines = md.splitlines()
    out: list[str] = []
    i = 0
    para: list[str] = []

    def flush_para():
        if para:
            out.append("<p>" + inline(" ".join(para)) + "</p>")
            para.clear()

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # fenced code block
        if stripped.startswith("```"):
            flush_para()
            i += 1
            buf = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            out.append("<pre><code>" + html.escape("\n".join(buf)) + "</code></pre>")
            continue

        # table
        if stripped.startswith("|") and i + 1 < len(lines) and is_separator(lines[i + 1]):
            flush_para()
            head = cells(stripped)
            aligns = []
            for c in cells(lines[i + 1]):
                aligns.append("right" if c.endswith(":") else
                              ("center" if c.startswith(":") and c.endswith(":") else "left"))
            i += 2
            body = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                body.append(cells(lines[i]))
                i += 1
            th = "".join(f'<th style="text-align:{aligns[j] if j < len(aligns) else "left"}">'
                         f"{inline(c)}</th>" for j, c in enumerate(head))
            trs = []
            for row in body:
                tds = "".join(
                    f'<td style="text-align:{aligns[j] if j < len(aligns) else "left"}">'
                    f"{inline(c)}</td>" for j, c in enumerate(row))
                trs.append(f"<tr>{tds}</tr>")
            out.append(f'<table><thead><tr>{th}</tr></thead><tbody>{"".join(trs)}'
                       "</tbody></table>")
            continue

        if not stripped:
            flush_para()
            i += 1
            continue

        if re.match(r"^---+$", stripped):
            flush_para()
            out.append("<hr/>")
            i += 1
            continue

        m = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if m:
            flush_para()
            lvl = len(m.group(1))
            out.append(f"<h{lvl}>{inline(m.group(2))}</h{lvl}>")
            i += 1
            continue

        if stripped.startswith("> "):
            flush_para()
            quote = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                quote.append(lines[i].strip().lstrip("> ").rstrip())
                i += 1
            out.append("<blockquote>" + inline(" ".join(quote)) + "</blockquote>")
            continue

        if re.match(r"^[-*]\s+", stripped):
            flush_para()
            items = []
            while i < len(lines) and re.match(r"^\s*[-*]\s+", lines[i]):
                items.append("<li>" + inline(re.sub(r"^\s*[-*]\s+", "", lines[i]).rstrip())
                             + "</li>")
                i += 1
            out.append("<ul>" + "".join(items) + "</ul>")
            continue

        # numbered lists are not used in this report; treat as paragraphs
        para.append(stripped)
        i += 1

    flush_para()
    return "\n".join(out)


CSS = """
@page { size: A4; margin: 17mm 15mm 18mm 15mm; }
* { box-sizing: border-box; }
body { margin: 0; color: #16202c; background: #fff;
  font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
  font-size: 10pt; line-height: 1.52; }
h1 { font-size: 19pt; line-height: 1.25; margin: 0 0 12px; color: #0f3057; }
h2 { font-size: 13.5pt; margin: 22px 0 9px; padding-bottom: 5px;
  border-bottom: 1px solid #ccd7e3; color: #0f3057; break-after: avoid; }
h3 { font-size: 11.5pt; margin: 16px 0 7px; color: #1b4f80; break-after: avoid; }
h4 { font-size: 10.5pt; margin: 13px 0 6px; color: #34495e; break-after: avoid; }
p { margin: 0 0 9px; }
strong { color: #0d2b4e; }
a { color: #185fa5; text-decoration: none; }
ul { margin: 4px 0 11px; padding-left: 19px; }
li { margin: 0 0 4px; }
hr { border: 0; border-top: 1px solid #ccd7e3; margin: 14px 0 20px; }
blockquote { margin: 10px 0 14px; padding: 8px 12px; background: #f2f7fc;
  border-left: 3px solid #2f6fb0; color: #274156; break-inside: avoid; }
code { font-family: Consolas, "Courier New", monospace; font-size: 8.6pt;
  background: #eef2f6; padding: 1px 3px; border-radius: 3px; color: #1d3557; }
pre { background: #f5f7fa; border: 1px solid #dce3eb; border-radius: 5px;
  padding: 9px 11px; margin: 8px 0 14px; white-space: pre-wrap; break-inside: avoid; }
pre code { background: none; padding: 0; font-size: 8.2pt; }
table { width: 100%; border-collapse: collapse; margin: 9px 0 15px;
  font-size: 8.5pt; break-inside: avoid; }
thead { display: table-header-group; }
tr { break-inside: avoid; }
th { background: #eaf1f8; color: #173f68; font-weight: 600; }
th, td { border: 1px solid #d5dfe9; padding: 5px 7px; vertical-align: top; }
tbody tr:nth-child(even) { background: #fafcfd; }
"""


def main() -> int:
    if not SRC.exists():
        raise SystemExit(f"missing {SRC}")
    chrome = next((p for p in CHROME_CANDIDATES if p.exists()), None)
    if chrome is None:
        raise SystemExit("Chrome/Edge not found; cannot render REPORT.pdf")

    body = md_to_html(SRC.read_text(encoding="utf-8"))
    doc = ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"/>"
           "<title>PS-3 Final Report</title><style>" + CSS + "</style></head><body>"
           + body + "</body></html>")
    tmp = Path(tempfile.gettempdir()) / "ps3_report_print.html"
    tmp.write_text(doc, encoding="utf-8")

    cmd = [str(chrome), "--headless=new", "--disable-gpu", "--no-sandbox",
           "--no-pdf-header-footer", "--virtual-time-budget=8000",
           f"--print-to-pdf={DST}", tmp.as_uri()]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if not DST.exists() or DST.stat().st_size < 5000:
        print(r.stdout[-1500:])
        print(r.stderr[-1500:])
        raise SystemExit("Chrome did not produce a usable PDF")
    print(f"wrote {DST} ({DST.stat().st_size:,} bytes) from {SRC.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
