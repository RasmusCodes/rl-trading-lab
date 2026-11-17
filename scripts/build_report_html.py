"""Render reports/REPORT.md into a single self-contained HTML file (figures embedded as base64).

    python scripts/build_report_html.py   -> reports/report.html
"""
from __future__ import annotations

import base64
import re
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports"

CSS = """
:root { color-scheme: light; --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
        --grid:#e1e0d9; --border:rgba(11,11,11,.10); --accent:#2a78d6; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { color-scheme: dark; --page:#0d0d0d;
        --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --grid:#2c2c2a; --border:rgba(255,255,255,.10);
        --accent:#3987e5; } }
:root[data-theme="dark"] { color-scheme: dark; --page:#0d0d0d; --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7;
        --grid:#2c2c2a; --border:rgba(255,255,255,.10); --accent:#3987e5; }
* { box-sizing: border-box; }
body { margin:0; background:var(--page); color:var(--ink); font: 15px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif; }
main { max-width: 920px; margin: 0 auto; padding: 32px 16px 80px; }
h1 { font-size: 28px; line-height:1.25; margin: 0 0 6px; }
h2 { font-size: 20px; margin: 40px 0 10px; padding-top: 8px; border-top: 1px solid var(--grid); }
h3 { font-size: 16px; margin: 28px 0 8px; }
p, li { color: var(--ink); }
em { color: var(--ink2); }
a { color: var(--accent); }
code { font-size: 13px; background: var(--surface); border: 1px solid var(--border); border-radius: 4px; padding: 0 4px; }
img { max-width: 100%; height: auto; display: block; margin: 14px 0; border: 1px solid var(--border);
      border-radius: 8px; background: #fcfcfb; }
table { border-collapse: collapse; width: 100%; font-size: 13px; margin: 12px 0 18px; display: block; overflow-x: auto; }
th, td { padding: 6px 8px; border-bottom: 1px solid var(--grid); text-align: left; white-space: nowrap; }
th { color: var(--ink2); font-weight: 600; }
td { font-variant-numeric: tabular-nums; }
blockquote { margin: 12px 0; padding: 8px 14px; border-left: 3px solid var(--accent); background: var(--surface); }
"""


def embed_images(html: str) -> str:
    def rep(m):
        src = m.group(1)
        p = (REP / src).resolve()
        if not p.exists():
            return m.group(0)
        b64 = base64.b64encode(p.read_bytes()).decode()
        return m.group(0).replace(src, f"data:image/png;base64,{b64}")

    return re.sub(r'<img[^>]*src="([^"]+\.png)"', rep, html)


def main():
    md_text = (REP / "REPORT.md").read_text(encoding="utf-8")
    md_text = re.sub(r"<!--.*?-->", "", md_text, flags=re.S)
    body = markdown.markdown(md_text, extensions=["tables", "fenced_code", "sane_lists"])
    body = embed_images(body)
    title = "RL for Trading Study"
    html = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
            f"content='width=device-width, initial-scale=1'><title>{title}</title><style>{CSS}</style></head>"
            f"<body><main>{body}</main></body></html>")
    (REP / "report.html").write_text(html, encoding="utf-8")
    print("wrote", REP / "report.html", f"{len(html) / 1e6:.1f} MB")


if __name__ == "__main__":
    sys.exit(main())
