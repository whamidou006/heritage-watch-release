#!/usr/bin/env python3
"""Render a Markdown document in this repository to PDF.

The PDFs are a convenience for sharing; Markdown in git stays the source of
truth. Re-render rather than editing a PDF.

Needs WeasyPrint, which needs Pango. Neither is a package dependency, because
nothing in the library or the student path uses them:

    conda create -p <env> -c conda-forge python=3.12 weasyprint
    <env>/bin/pip install markdown

    <env>/bin/python scripts/make_pdf.py research/REPORT.md -o out/REPORT.pdf

Links to other files in the repository are rewritten to the GitHub blob URL of
the branch you pass with --branch, so they still resolve once the PDF has left
the working tree.
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
import subprocess
from pathlib import Path

import markdown
from weasyprint import HTML

REPO = "https://github.com/whamidou006/heritage-watch-release"

# A4. The font stacks name only faces that ship COMPLETE in the rendering
# environment. Two traps, both of which fail silently rather than erroring:
# naming an absent family ("DejaVu Serif") renders the whole body in
# monospace, and naming a family with no bold cut (the bundled DejaVu Sans is
# regular-only) drops every **emphasis** in the document. Ubuntu and Source
# Code Pro ship regular/bold/italic/bold-italic, so both survive. DejaVu Sans
# is kept last for its wider glyph coverage (arrows, ·, ±).
#
# Tables are the reason this stylesheet exists at all: the reports are mostly
# tables, and the default breaks them across pages without repeating the
# header.
CSS = """
@page {
  size: A4; margin: 20mm 18mm 18mm 18mm;
  @bottom-center { content: counter(page) " / " counter(pages);
                   font: 8.5pt 'Ubuntu', 'DejaVu Sans', sans-serif; color: #777; }
}
@page :first { @bottom-center { content: ""; } }
body { font: 10pt/1.55 'Ubuntu', 'DejaVu Sans', sans-serif; color: #1a1a1a;
       hyphens: auto; text-align: justify; }
h1 { font: bold 20pt 'Ubuntu', 'DejaVu Sans', sans-serif; margin: 0 0 4pt; line-height: 1.25;
     text-align: left; }
h2 { font: bold 13.5pt 'Ubuntu', 'DejaVu Sans', sans-serif; margin: 20pt 0 6pt;
     padding-bottom: 3pt; border-bottom: 1px solid #d0d0d0;
     text-align: left; break-after: avoid; }
h3 { font: bold 11pt 'Ubuntu', 'DejaVu Sans', sans-serif; margin: 14pt 0 4pt; text-align: left;
     break-after: avoid; }
h4 { font: bold 10pt 'Ubuntu', 'DejaVu Sans', sans-serif; margin: 11pt 0 3pt; text-align: left;
     break-after: avoid; }
p, li { orphans: 2; widows: 2; }
a { color: #0b5cad; text-decoration: none; }

code { font: 8.6pt 'Source Code Pro', 'Ubuntu Mono', monospace; background: #f4f4f4;
       padding: 0.5pt 2pt; border-radius: 2px; }
pre { font-size: 8.4pt; background: #f7f7f7; border: 1px solid #e3e3e3;
      border-left: 3px solid #b8b8b8; border-radius: 3px;
      padding: 6pt 8pt; line-height: 1.35; white-space: pre-wrap;
      word-wrap: break-word; break-inside: avoid; text-align: left; }
pre code { background: none; padding: 0; font-size: inherit; }

table { border-collapse: collapse; width: 100%; margin: 8pt 0;
        font: 8.6pt 'Ubuntu', 'DejaVu Sans', sans-serif; break-inside: auto; }
thead { display: table-header-group; }          /* repeat header across pages */
tr { break-inside: avoid; }
th, td { border: 1px solid #dcdcdc; padding: 3pt 5pt; text-align: left;
         vertical-align: top; }
th { background: #f0f0f0; font-weight: bold; }
tbody tr:nth-child(even) { background: #fafafa; }

/* GitHub's idiom for a caption is <sub>. Rendered as a true subscript it
   wrecks the baseline of the paragraph it sits in. */
sub.caption, sub { font-size: 8.8pt; vertical-align: baseline; color: #555; }
sup { font-size: 7.5pt; }

blockquote { margin: 8pt 0; padding: 4pt 10pt; border-left: 3px solid #c9c9c9;
             background: #fbfbfb; color: #444; }
hr { border: none; border-top: 1px solid #ddd; margin: 14pt 0; }
img { max-width: 100%; }

.subtitle { font: 8.6pt 'Ubuntu', 'DejaVu Sans', sans-serif; color: #666; margin: 0 0 14pt;
            padding-bottom: 8pt; border-bottom: 2px solid #333; }
"""


def git(*args: str, cwd: Path) -> str:
    return subprocess.run(["git", *args], cwd=cwd, text=True,
                          capture_output=True).stdout.strip()


def absolutise_links(md: str, root: Path, src: Path, branch: str) -> str:
    """Rewrite repo-relative links so they still work outside the checkout."""
    base = f"{REPO}/blob/{branch}"

    def repl(m: re.Match) -> str:
        text, target = m.group(1), m.group(2)
        if re.match(r"^(https?:|mailto:|#)", target):
            return m.group(0)
        anchor = ""
        if "#" in target:
            target, anchor = target.split("#", 1)
            anchor = "#" + anchor
        if not target:
            return m.group(0)
        resolved = (src.parent / target).resolve()
        try:
            rel = resolved.relative_to(root.resolve())
        except ValueError:
            return m.group(0)
        return f"[{text}]({base}/{rel}{anchor})"

    return re.sub(r"\[([^\]]*)\]\(([^)]+)\)", repl, md)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", type=Path, help="Markdown file to render")
    ap.add_argument("-o", "--out", type=Path, help="output PDF (default: out/<name>.pdf)")
    ap.add_argument("--title", help="override the document title")
    ap.add_argument("--branch", help="branch the links should point at (default: current)")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    src = args.source.resolve()
    if not src.exists():
        raise SystemExit(f"{src} does not exist")
    out = args.out or root / "out" / (src.stem + ".pdf")
    out.parent.mkdir(parents=True, exist_ok=True)

    branch = args.branch or git("rev-parse", "--abbrev-ref", "HEAD", cwd=root) or "main"
    commit = git("rev-parse", "--short", "HEAD", cwd=root)
    dirty = bool(git("status", "--porcelain", cwd=root))

    text = src.read_text(encoding="utf-8")
    text = absolutise_links(text, root, src, branch)

    # Take the title from the first H1 and drop it, so the rendered heading is
    # not printed twice.
    title = args.title
    lines = text.splitlines()
    if lines and lines[0].startswith("# "):
        title = title or lines[0][2:].strip()
        lines = lines[1:]
    text = "\n".join(lines).lstrip("\n")

    body = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists",
                                               "attr_list", "md_in_html"])

    stamp = dt.date.today().isoformat()
    rel = src.relative_to(root) if src.is_relative_to(root) else src.name
    provenance = (f"{rel} · branch {branch} · commit {commit}"
                  f"{' + uncommitted changes' if dirty else ''} · rendered {stamp}")

    html = (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>{title or rel}</title></head><body>"
            f"<h1>{title or rel}</h1>"
            f"<p class='subtitle'>{provenance}<br>{REPO}</p>"
            f"{body}</body></html>")

    HTML(string=html, base_url=str(src.parent)).write_pdf(out, stylesheets=[__css()])
    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} KB)")


def __css():
    from weasyprint import CSS as _CSS
    return _CSS(string=CSS)


if __name__ == "__main__":
    main()
