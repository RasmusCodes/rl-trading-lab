"""Build the shareable report page (single HTML file, figures embedded) from reports/REPORT.md.

    python scripts/build_report_artifact.py   -> reports/rl-study-page.html
"""
from __future__ import annotations

import base64
import html as htmlmod
import re
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports"

CAPTIONS = {
    "controls.png": "Test Sharpe on three synthetic markets: a random walk and two with a planted edge (3-seed "
                    "ensembles, 95% bootstrap intervals; reference line: buy & hold).",
    "gamma.png": "Discount-factor sweep on the planted-edge controls. PPO is indifferent; the bootstrapping agents "
                 "are erratic but better below 0.95.",
    "baselines_spy.png": "SPY baselines through the same environment, test blocks 2007-2018 chained (95% bootstrap "
                         "intervals; shaded band: random strategies).",
    "main_bars.png": "Nine RL agents on SPY: Sharpe of each 3-seed ensemble with its 95% interval, against buy & hold.",
    "main_heatmap.png": "Test Sharpe by 2-year block (mean of 3 seeds), with buy & hold and the 50/200-day trend "
                        "filter for comparison.",
    "main_equity.png": "Chained test blocks 2007-2018: the three best agent ensembles against buy & hold and the "
                       "best baseline (each block restarts flat).",
    "val_vs_test.png": "Left: validation Sharpe of the selected checkpoint against the next two years' test Sharpe. "
                       "Right: validation-selected against last checkpoint.",
    "training_curves.png": "Validation Sharpe during training, fold 3, three seeds per algorithm.",
    "rewards.png": "Reward functions: out-of-sample Sharpe with exposure and turnover.",
    "hacking.png": "Reward-hacking variants: penalties against exposure, risk stops against episode length, and "
                   "objective variants against test Sharpe.",
    "features.png": "State representation: feature sets against Sharpe, exposure and turnover.",
    "randomized.png": "Training-environment variants evaluated on SPY.",
    "hpo.png": "PPO hyperparameter search: the original search (flawed metric) and the repeat after the fix.",
    "leakage.png": "One planted look-ahead feature at a time: mean test Sharpe of PPO runs.",
    "portfolio_etf9.png": "Nine-ETF portfolio, test 2011-2018: RL seed ensembles (blue) against baselines (gray).",
    "futures_fut4.png": "Four futures, test 2011-2018: RL seed ensembles (blue) against baselines (gray).",
    "equities_eq10.png": "Ten end-2000 large caps with pooled agents, test 2007-2018.",
    "robustness.png": "Re-scored checkpoints: Sharpe against cost multiplier and against capital.",
    "holdout.png": "Final holdout, 2019-01 to 2026-11, evaluated once: RL seed ensembles (blue) against baselines (gray).",
}

TIMING = """
<figure class="timing" aria-label="Execution timing">
  <div class="tick"><span class="when">close t</span><span class="what">observe bars up to t and the book marked at the close; choose target weights</span></div>
  <div class="tick"><span class="when">open t+1</span><span class="what">old position carries the overnight gap; fill at the open plus spread, slippage, impact, commission</span></div>
  <div class="tick"><span class="when">close t+1</span><span class="what">mark the new position; interest, borrow; the reward for the step</span></div>
</figure>
"""

STRIP = """
<section class="strip" aria-label="Findings at a glance">
  <div class="stat"><span class="num">0.19 to 0.53</span><span class="lbl">nine RL agents, SPY 2007-2018 (Sharpe)</span></div>
  <div class="stat"><span class="num">0.41</span><span class="lbl">buy &amp; hold, same windows; no agent different at p &lt; 0.39</span></div>
  <div class="stat"><span class="num">&minus;0.03</span><span class="lbl">best information ratio against buy &amp; hold among 72 strategies</span></div>
  <div class="stat"><span class="num">0.77 vs 0.67</span><span class="lbl">holdout 2019-2026: buy &amp; hold vs the best agent</span></div>
</section>
"""

CSS = r"""
/* Layout: one serif reading column (~68ch) with a sticky contents rail on wide screens; figures sit on light plates. */
:root {
  --bg: #f6f7f9;
  --surface: #ffffff;
  --ink: #141a24;
  --ink-2: #485263;
  --muted: #7a8494;
  --rule: #dde2ea;
  --accent: #2a78d6;
  --link: #1c5cab;
  --plate: #fcfcfb;
  --plate-edge: #e4e7ec;
  --code-bg: #eef1f5;
  --f-head: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --f-body: "Source Serif 4", Georgia, "Times New Roman", serif;
  --f-mono: "IBM Plex Mono", ui-monospace, "Cascadia Mono", Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #0f1319; --surface: #161b23; --ink: #e7ebf1; --ink-2: #b4becb; --muted: #8b95a4; --rule: #273141;
    --accent: #5a9ff0; --link: #8bbcf6; --plate: #f2f3f4; --plate-edge: #2f3948; --code-bg: #1b222c;
    color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --bg: #0f1319; --surface: #161b23; --ink: #e7ebf1; --ink-2: #b4becb; --muted: #8b95a4; --rule: #273141;
  --accent: #5a9ff0; --link: #8bbcf6; --plate: #f2f3f4; --plate-edge: #2f3948; --code-bg: #1b222c;
  color-scheme: dark;
}
* { box-sizing: border-box; }
html { scroll-padding-top: 16px; }
body {
  background: var(--bg); color: var(--ink); font-family: var(--f-body); font-size: 17px; line-height: 1.62;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px));
  padding-block: 0 72px;
}
.page { max-width: 1180px; margin: 0 auto; display: grid; grid-template-columns: minmax(0, 1fr); gap: 40px; }
@media (min-width: 1080px) { .page { grid-template-columns: 220px minmax(0, 1fr); } }

header.masthead { max-width: 1180px; margin: 0 auto; padding-block: 40px 20px; border-bottom: 1px solid var(--rule); }
.eyebrow { font-family: var(--f-head); font-size: 12.5px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }
h1 { font-family: var(--f-head); font-weight: 700; font-size: clamp(28px, 4.2vw, 42px); line-height: 1.12; letter-spacing: -0.01em;
     margin: 10px 0 8px; text-wrap: balance; max-width: 22ch; }
.dek { font-family: var(--f-head); color: var(--ink-2); font-size: 15px; margin: 0; }

.strip { max-width: 1180px; margin: 22px auto 8px; display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1px;
         background: var(--rule); border: 1px solid var(--rule); border-radius: 10px; overflow: hidden; }
.stat { background: var(--surface); padding: 14px 16px; display: flex; flex-direction: column; gap: 4px; }
.stat .num { font-family: var(--f-head); font-weight: 600; font-size: 24px; letter-spacing: -0.01em; color: var(--ink); }
.stat .lbl { font-family: var(--f-head); font-size: 13px; line-height: 1.35; color: var(--ink-2); }

nav.toc { font-family: var(--f-head); font-size: 13.5px; line-height: 1.4; }
nav.toc .toc-title { font-size: 12px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); margin: 0 0 10px; }
nav.toc ol { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 6px; }
nav.toc ol ol { margin: 6px 0 2px 12px; gap: 4px; font-size: 12.5px; }
nav.toc a { color: var(--ink-2); text-decoration: none; }
nav.toc a:hover { color: var(--link); text-decoration: underline; }
@media (min-width: 1080px) {
  nav.toc { position: sticky; top: calc(env(safe-area-inset-top, 0px) + 24px); align-self: start; max-height: calc(100vh - 48px);
            overflow-y: auto; padding-top: 28px; }
  nav.toc details > summary { display: none; }
}
@media (max-width: 1079px) {
  nav.toc { border: 1px solid var(--rule); border-radius: 10px; background: var(--surface); padding: 10px 14px; margin-top: 16px; }
  nav.toc details > summary { cursor: pointer; font-size: 14px; color: var(--ink); }
  nav.toc .toc-title { display: none; }
  nav.toc ol { margin-top: 10px; }
}

article { min-width: 0; max-width: 72ch; padding-top: 16px; }
article h2 { font-family: var(--f-head); font-weight: 700; font-size: 25px; line-height: 1.2; margin: 52px 0 12px; text-wrap: balance;
             padding-top: 14px; border-top: 1px solid var(--rule); }
article h3 { font-family: var(--f-head); font-weight: 600; font-size: 19px; line-height: 1.3; margin: 34px 0 8px; text-wrap: balance; }
article p, article li { margin: 0 0 12px; }
article ul, article ol { padding-left: 22px; margin: 0 0 14px; }
article li > ul, article li > ol { margin-top: 6px; }
article strong { font-weight: 600; }
article a { color: var(--link); }
article code { font-family: var(--f-mono); font-size: 0.84em; background: var(--code-bg); padding: 1px 5px; border-radius: 4px; }
article pre { background: var(--code-bg); border: 1px solid var(--rule); border-radius: 8px; padding: 12px 14px; overflow-x: auto; }
article pre code { background: none; padding: 0; font-size: 13px; line-height: 1.5; }

.table-wrap { overflow-x: auto; margin: 14px 0 20px; border: 1px solid var(--rule); border-radius: 8px; background: var(--surface); }
table { border-collapse: collapse; width: 100%; font-family: var(--f-head); font-size: 13.5px; line-height: 1.4; font-variant-numeric: tabular-nums; }
th, td { padding: 8px 10px; border-bottom: 1px solid var(--rule); text-align: left; vertical-align: top; }
th { color: var(--ink-2); font-weight: 600; background: var(--surface); white-space: nowrap; }
tr:last-child td { border-bottom: none; }

figure.plate { margin: 22px 0 26px; }
@media (min-width: 1080px) { figure.plate { width: min(940px, calc(100vw - 340px)); } }
figure.plate .img { background: var(--plate); border: 1px solid var(--plate-edge); border-radius: 10px; padding: 10px; }
figure.plate img { display: block; width: 100%; height: auto; max-width: 100%; }
figcaption { font-family: var(--f-head); font-size: 13.5px; line-height: 1.45; color: var(--ink-2); margin-top: 8px; max-width: 80ch; }
figcaption b { color: var(--ink); font-weight: 600; }

figure.timing { margin: 16px 0 20px; display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 0; position: relative;
                font-family: var(--f-head); }
figure.timing::before { content: ""; position: absolute; left: 6px; right: 6px; top: 6px; height: 2px; background: var(--rule); }
figure.timing .tick { position: relative; padding: 22px 12px 0 0; display: flex; flex-direction: column; gap: 4px; }
figure.timing .tick::before { content: ""; position: absolute; left: 0; top: 0; width: 14px; height: 14px; border-radius: 50%;
                              background: var(--accent); border: 3px solid var(--bg); }
figure.timing .when { font-family: var(--f-mono); font-size: 13px; color: var(--ink); font-weight: 600; }
figure.timing .what { font-size: 13px; line-height: 1.4; color: var(--ink-2); }
@media (max-width: 560px) {
  figure.timing { grid-template-columns: minmax(0, 1fr); gap: 14px; }
  figure.timing::before { left: 6px; right: auto; top: 6px; bottom: 6px; width: 2px; height: auto; }
  figure.timing .tick { padding: 0 0 0 26px; }
}
footer.colophon { max-width: 1180px; margin: 56px auto 0; padding-top: 16px; border-top: 1px solid var(--rule);
                  font-family: var(--f-head); font-size: 13px; color: var(--muted); }
a:focus-visible, summary:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; border-radius: 3px; }
@media (prefers-reduced-motion: reduce) { html { scroll-behavior: auto; } }
@media (prefers-reduced-motion: no-preference) { html { scroll-behavior: smooth; } }
"""


def toc_html(tokens) -> str:
    def items(toks, depth):
        out = []
        for t in toks:
            if t["level"] == 1:
                out.append(items(t.get("children", []), depth))
                continue
            sub = ""
            if t.get("children") and t["level"] == 2 and t["name"].startswith("7."):
                sub = "<ol>" + "".join(f'<li><a href="#{c["id"]}">{c["name"]}</a></li>' for c in t["children"]) + "</ol>"
            out.append(f'<li><a href="#{t["id"]}">{t["name"]}</a>{sub}</li>')
        return "".join(out)

    return ('<nav class="toc" aria-label="Contents"><details open><summary>Contents</summary>'
            '<p class="toc-title">Contents</p><ol>' + items(tokens, 0) + "</ol></details></nav>")


def main():
    text = (REP / "REPORT.md").read_text(encoding="utf-8")
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    md = markdown.Markdown(extensions=["tables", "fenced_code", "sane_lists", "toc"],
                           extension_configs={"toc": {"permalink": False}})
    body = md.convert(text)
    tokens = md.toc_tokens
    # the h1 and its italic dek go into the masthead
    m = re.search(r"<h1[^>]*>(.*?)</h1>\s*<p><em>(.*?)</em></p>", body, flags=re.S)
    title_h1, dek = (m.group(1), m.group(2)) if m else ("RL trading study", "")
    body = body[: m.start()] + body[m.end():] if m else body
    # figures: numbered, captioned, embedded
    n = [0]

    def fig(mm):
        alt, src = mm.group(1), mm.group(2)
        p = REP / src
        n[0] += 1
        cap = CAPTIONS.get(p.name, alt)
        b64 = base64.b64encode(p.read_bytes()).decode()
        return (f'<figure class="plate"><div class="img"><img src="data:image/png;base64,{b64}" '
                f'alt="Figure {n[0]}. {htmlmod.escape(cap)}" loading="lazy"></div>'
                f"<figcaption><b>Figure {n[0]}.</b> {htmlmod.escape(cap)}</figcaption></figure>")

    body = re.sub(r'<p><img alt="([^"]*)" src="([^"]+\.png)"\s*/?></p>', fig, body)
    body = body.replace("<table>", '<div class="table-wrap"><table>').replace("</table>", "</table></div>")
    # execution-timing strip after the "Timing." paragraph of section 3
    body = re.sub(r"(<p><strong>Timing\.</strong>.*?</p>)", lambda mm: mm.group(1) + TIMING, body, count=1, flags=re.S)
    page = f"""<title>Can RL Beat Buy-and-Hold?</title>
<meta name="claude:viewport-fit-sides" content="cover">
<meta name="description" content="Walk-forward study of nine reinforcement-learning algorithms against simple baselines.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;600;700&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&display=swap">
<style>{CSS}</style>
<header class="masthead">
  <div class="eyebrow">Research report · RL Trading Lab · data through 12 Nov 2026</div>
  <h1>{title_h1}</h1>
  <p class="dek">{dek}</p>
</header>
{STRIP}
<div class="page">
{toc_html(tokens)}
<article>
{body}
</article>
</div>
<footer class="colophon">Every figure is generated from the lab's run directories by <code>scripts/build_report.py</code>.
Sharpe ratios are on returns in excess of T-bills, after all trading costs. Nothing here is investment advice.</footer>
"""
    out = REP / "rl-study-page.html"
    out.write_text(page, encoding="utf-8")
    print("wrote", out, f"{len(page) / 1e6:.2f} MB, {n[0]} figures")


if __name__ == "__main__":
    sys.exit(main())
