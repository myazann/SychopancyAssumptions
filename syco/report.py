"""The self-contained HTML report `syco analyze` writes beside its tables.

The CSVs are the record; this is the thing a collaborator opens. It carries the
summary before the detail -- three verdict cards, one per question, whose state
comes from the corrected p-values rather than from a judgment call -- and then
the tables and figures behind each.

Everything is inlined: the data as JSON, the styles and the script in the
document. The page therefore renders from a file:// path, an email attachment,
or a published artifact with no server and no external fetch.

Nothing here is an image. Charts are drawn from the tokens below, so they
follow the viewer's theme instead of needing a light and a dark rendering of
every one, their numbers stay selectable, and the reader can change what is
plotted rather than scrolling past every chart the pipeline could make.
"""
from __future__ import annotations

import html

import numpy as np
import pandas as pd


#: Verdict states. `detected` and `none` are read off a corrected p-value;
#: `descriptive` marks a table that was never a test -- the top-5 lists and the
#: marked words, which describe language and have no null behind them.
def esc(text) -> str:
    """HTML-escape a value for interpolation into a page."""
    return html.escape(str(text), quote=True)


_esc = esc


def _data_uri(path: Path) -> str:
    return ("data:image/png;base64,"
            + base64.b64encode(Path(path).read_bytes()).decode("ascii"))


def _format(value, column: str = "") -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return '<span class="nil">-</span>'
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if column.startswith(("p_", "q_")) or column.endswith(("_value", "_p")):
            # An asymptotic chi-square p can underflow to exactly 0 in float64;
            # printing "0.000" claims a certainty the arithmetic does not have.
            if value == 0.0:
                return "&lt;1e-308"
            return "&lt;0.001" if 0 < value < 0.001 else f"{value:.3f}"
        if abs(value) >= 1000 or (value and abs(value) < 1e-3):
            return f"{value:.3g}"
        return f"{value:,.3f}"
    return _esc(value)


def table_html(frame: pd.DataFrame, *, caption: str = "",
               numeric_columns=None) -> str:
    """A scrollable table. Digits are tabular so columns of numbers line up."""
    if frame is None or frame.empty:
        return f'<p class="empty">{_esc(caption)}: nothing to show.</p>'
    numeric_columns = set(numeric_columns or [
        c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])])
    head = "".join(
        f'<th class="{"num" if c in numeric_columns else ""}">{_esc(c)}</th>'
        for c in frame.columns)
    body = []
    for row in frame.itertuples(index=False, name=None):
        cells = "".join(
            f'<td class="{"num" if c in numeric_columns else ""}">'
            f'{_format(v, str(c))}</td>'
            for c, v in zip(frame.columns, row))
        body.append(f"<tr>{cells}</tr>")
    caption_html = f"<figcaption>{_esc(caption)}</figcaption>" if caption else ""
    return (
        '<figure class="tablewrap">'
        f'<div class="scroll"><table><thead><tr>{head}</tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table></div>'
        f'{caption_html}</figure>'
    )


def stat(value: str, label: str, note: str = "") -> str:
    return (f'<div class="stat"><div class="stat-value">{_esc(value)}</div>'
            f'<div class="stat-label">{_esc(label)}</div>'
            + (f'<div class="stat-note">{_esc(note)}</div>' if note else "")
            + "</div>")


STYLE = """
:root {
  color-scheme: light;
  --ground:#f5f7f8; --surface:#ffffff; --well:#eef1f4;
  --ink:#161a1f; --ink-2:#4c5661; --ink-3:#78838f;
  --rule:#dde2e7; --rule-strong:#c3ccd4;
  --accent:#1e4f8f; --accent-soft:#e6eef8; --accent-ink:#1e4f8f;
  --ok:#0b7d2f; --ok-soft:#e2f2e6;
  --null-tone:#6b7682; --null-soft:#eceff2;
  --caution:#94620d; --caution-soft:#f7eeda;
  --shadow:0 1px 2px rgba(20,26,33,.05), 0 8px 24px -18px rgba(20,26,33,.35);
  --term-here:#2a78d6; --term-there:#eb6834;
  --series-1:#2a78d6; --series-2:#eb6834; --series-3:#1baf7a;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --ground:#0e1013; --surface:#16191d; --well:#1e232a;
    --ink:#e9ecef; --ink-2:#a6b0bb; --ink-3:#79838f;
    --rule:#272d34; --rule-strong:#39414a;
    --accent:#7fadea; --accent-soft:#182432; --accent-ink:#9cc2f2;
    --ok:#4fbf6a; --ok-soft:#15291a;
    --null-tone:#8c96a1; --null-soft:#1d2228;
    --caution:#dda93f; --caution-soft:#2b2415;
    --shadow:0 1px 2px rgba(0,0,0,.4), 0 10px 30px -20px rgba(0,0,0,.8);
    --term-here:#3987e5; --term-there:#d95926;
    --series-1:#3987e5; --series-2:#d95926; --series-3:#199e70;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --ground:#0e1013; --surface:#16191d; --well:#1e232a;
  --ink:#e9ecef; --ink-2:#a6b0bb; --ink-3:#79838f;
  --rule:#272d34; --rule-strong:#39414a;
  --accent:#7fadea; --accent-soft:#182432; --accent-ink:#9cc2f2;
  --ok:#4fbf6a; --ok-soft:#15291a;
  --null-tone:#8c96a1; --null-soft:#1d2228;
  --caution:#dda93f; --caution-soft:#2b2415;
  --shadow:0 1px 2px rgba(0,0,0,.4), 0 10px 30px -20px rgba(0,0,0,.8);
  --term-here:#3987e5; --term-there:#d95926;
  --series-1:#3987e5; --series-2:#d95926; --series-3:#199e70;
}

*,*::before,*::after { box-sizing:border-box; }
body {
  margin:0; background:var(--ground); color:var(--ink);
  font-family:"IBM Plex Serif", Georgia, "Times New Roman", serif;
  font-size:16.5px; line-height:1.65;
  -webkit-font-smoothing:antialiased;
}
.wrap { max-width:1120px; margin:0 auto; padding:0 28px 96px; }
.prose { max-width:68ch; }

h1,h2,h3,h4,.chip,.stat-label,.eyebrow,th,.verdict-index,nav.index a {
  font-family:"IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
}
h1 { font-size:clamp(2rem,4.2vw,2.9rem); line-height:1.08; margin:0 0 .35em;
     font-weight:600; letter-spacing:-.022em; text-wrap:balance; }
h2 { font-size:1.55rem; font-weight:600; letter-spacing:-.014em; margin:0 0 .2em;
     text-wrap:balance; }
h3 { font-size:1.06rem; font-weight:600; letter-spacing:-.006em; margin:0 0 .3em; }
h4 { font-size:.9rem; font-weight:600; margin:2.4em 0 .6em; color:var(--ink-2);
     letter-spacing:.01em; }
p { margin:0 0 1em; }
a { color:var(--accent-ink); text-decoration-thickness:1px;
    text-underline-offset:2px; }
a:focus-visible, .verdict:focus-visible {
  outline:2px solid var(--accent); outline-offset:3px; border-radius:4px; }

.eyebrow {
  font-size:.72rem; font-weight:600; letter-spacing:.1em; text-transform:uppercase;
  color:var(--ink-3); margin:0 0 .9em;
}

/* --- masthead ------------------------------------------------------- */
header.masthead {
  border-bottom:1px solid var(--rule); background:var(--surface);
  padding:clamp(38px,6vw,68px) 0 34px; margin-bottom:38px;
}
header.masthead .wrap { padding-bottom:0; }
.lede { font-size:1.12rem; color:var(--ink-2); max-width:64ch; margin:0 0 1.4em; }
.runline {
  font-family:"IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size:.76rem; color:var(--ink-3); line-height:1.9;
  border-top:1px solid var(--rule); padding-top:14px; margin-top:26px;
  display:flex; flex-wrap:wrap; gap:6px 22px;
}
.runline b { font-weight:500; color:var(--ink-2); }

/* --- verdict cards --------------------------------------------------- */
.verdicts { display:grid; gap:14px; grid-template-columns:repeat(auto-fit,minmax(258px,1fr));
            margin:0 0 46px; }
.verdict {
  display:block; position:relative; background:var(--surface); color:inherit;
  border:1px solid var(--rule); border-radius:10px; padding:20px 20px 18px 20px;
  text-decoration:none; box-shadow:var(--shadow);
  transition:border-color .15s ease, transform .15s ease;
}
.verdict:hover { border-color:var(--rule-strong); transform:translateY(-1px); }
.verdict-index {
  position:absolute; top:18px; right:18px; font-size:1.5rem; font-weight:600;
  color:var(--rule-strong); line-height:1;
}
.verdict h3 { margin:.65em 0 .3em; padding-right:26px; }
.verdict-headline { margin:0 0 .45em; font-size:.98rem; color:var(--ink); }
.verdict-detail { margin:0; font-size:.85rem; color:var(--ink-3); line-height:1.5; }

.chip {
  display:inline-flex; align-items:center; gap:6px;
  font-size:.7rem; font-weight:600; letter-spacing:.05em; text-transform:uppercase;
  padding:3px 9px; border-radius:999px; white-space:nowrap;
}
.chip::before { content:""; width:6px; height:6px; border-radius:50%;
                background:currentColor; flex:none; }
.chip-ok { color:var(--ok); background:var(--ok-soft); }
.chip-null { color:var(--null-tone); background:var(--null-soft); }
.chip-caution { color:var(--caution); background:var(--caution-soft); }

/* --- sections -------------------------------------------------------- */
section.finding { margin:0 0 74px; scroll-margin-top:22px; }
.section-head {
  display:flex; align-items:baseline; gap:16px; border-top:2px solid var(--ink);
  padding-top:16px; margin-bottom:20px;
}
.section-number {
  font-family:"IBM Plex Mono", ui-monospace, monospace; font-size:.92rem;
  color:var(--ink-3); font-weight:500; flex:none;
}

.stats { display:flex; flex-wrap:wrap; gap:10px; margin:26px 0 30px; }
.stat {
  background:var(--well); border-radius:8px; padding:13px 17px; min-width:132px;
  flex:1 1 132px;
}
.stat-value { font-size:1.5rem; font-weight:600; line-height:1.15;
              font-family:"IBM Plex Sans", system-ui, sans-serif; }
.stat-label { font-size:.72rem; font-weight:600; letter-spacing:.055em;
              text-transform:uppercase; color:var(--ink-3); margin-top:4px; }
.stat-note { font-size:.78rem; color:var(--ink-2); margin-top:5px; line-height:1.4;
             font-family:"IBM Plex Sans", system-ui, sans-serif; }

/* --- figures and tables ---------------------------------------------- */
figure { margin:26px 0; }

figcaption { font-size:.82rem; color:var(--ink-3); margin-top:9px; line-height:1.5;
             max-width:76ch; }

.scroll { overflow-x:auto; border:1px solid var(--rule); border-radius:8px;
          background:var(--surface); }
table { border-collapse:collapse; width:100%; font-size:.82rem;
        font-family:"IBM Plex Mono", ui-monospace, SFMono-Regular, monospace; }
thead th {
  position:sticky; top:0; background:var(--well); color:var(--ink-2);
  font-size:.7rem; font-weight:600; letter-spacing:.045em; text-transform:uppercase;
  text-align:left; padding:9px 13px; white-space:nowrap;
  border-bottom:1px solid var(--rule-strong);
}
td { padding:7px 13px; border-bottom:1px solid var(--rule); white-space:nowrap;
     color:var(--ink-2); }
tbody tr:last-child td { border-bottom:none; }
tbody tr:hover td { background:var(--well); }
td.num, th.num { text-align:right; font-variant-numeric:tabular-nums; }
.nil { color:var(--ink-3); }
.empty { color:var(--ink-3); font-size:.9rem; }

/* --- notes ----------------------------------------------------------- */
.note {
  background:var(--accent-soft); border-left:3px solid var(--accent);
  border-radius:0 8px 8px 0; padding:15px 20px; margin:26px 0;
  font-size:.94rem; color:var(--ink-2);
}
.note strong { color:var(--ink); }
.note p:last-child { margin-bottom:0; }
.note.caveat { background:var(--caution-soft); border-left-color:var(--caution); }

/* --- the explorer ---------------------------------------------------- */
.controls {
  display:flex; flex-wrap:wrap; gap:12px 22px; align-items:flex-end;
  background:var(--well); border-radius:10px; padding:16px 20px; margin:26px 0 22px;
}
.controls label, .controls .control {
  display:flex; flex-direction:column; gap:5px;
  font-family:"IBM Plex Sans", system-ui, sans-serif;
  font-size:.72rem; font-weight:600; letter-spacing:.055em;
  text-transform:uppercase; color:var(--ink-3);
}
.controls select {
  font-family:"IBM Plex Sans", system-ui, sans-serif;
  font-size:.92rem; font-weight:400; letter-spacing:0; text-transform:none;
  color:var(--ink); background:var(--surface);
  border:1px solid var(--rule-strong); border-radius:6px; padding:7px 10px;
  min-width:190px; max-width:100%;
}
.controls select:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }

.panes { display:grid; gap:26px; grid-template-columns:repeat(auto-fit,minmax(330px,1fr)); }
.pane { min-width:0; }
.pane h3 { margin:0 0 .25em; }
.pane-note { font-size:.82rem; color:var(--ink-3); margin:0 0 14px; line-height:1.5;
             min-height:3.2em; }

.legend { display:flex; flex-wrap:wrap; align-items:center; gap:6px 14px;
          font-size:.78rem; color:var(--ink-2); margin:-6px 0 14px;
          font-family:"IBM Plex Sans", system-ui, sans-serif; }
.key { width:10px; height:10px; border-radius:2px; display:inline-block;
       margin-right:-8px; }
.key-here { background:var(--term-here); }
.key-there { background:var(--term-there); }

.bars { display:flex; flex-direction:column; gap:3px; }
.bar-row {
  display:grid; grid-template-columns:minmax(96px,34%) 1fr 52px;
  align-items:center; gap:10px; padding:2px 4px; border-radius:4px;
}
.bar-row:hover { background:var(--well); }
.bar-term {
  font-family:"IBM Plex Mono", ui-monospace, monospace; font-size:.79rem;
  color:var(--ink); overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
}
.bar-track { position:relative; height:13px; display:flex; }
.bar-row.diverging .bar-track { justify-content:center; }
.bar-fill { background:var(--term-here); border-radius:3px; height:100%;
            min-width:2px; }
.bar-row.diverging .bar-fill { position:absolute; left:50%; }
.bar-row.diverging .bar-fill.negative { left:auto; right:50%;
                                        background:var(--term-there); }
.bar-row.diverging .bar-track::before {
  content:""; position:absolute; left:50%; top:-2px; bottom:-2px;
  border-left:1px solid var(--rule-strong);
}
.bar-value {
  font-family:"IBM Plex Mono", ui-monospace, monospace; font-size:.75rem;
  color:var(--ink-2); text-align:right; font-variant-numeric:tabular-nums;
}
.small { font-size:.84rem; color:var(--ink-3); }

/* --- the persona chart and its findings ------------------------------ */
.chart { margin:22px 0 26px; }
.mini { display:flex; flex-direction:column; gap:5px; }
.mini-row { display:grid; grid-template-columns:minmax(88px,20%) 1fr;
            align-items:center; gap:12px; }
.mini-label {
  font-family:"IBM Plex Mono", ui-monospace, monospace; font-size:.78rem;
  color:var(--ink-2); text-align:right;
}
.mini-track { position:relative; height:16px; }
.mini-track::before {
  content:""; position:absolute; left:50%; top:0; bottom:0;
  border-left:1px solid var(--rule-strong);
}
.mini-bar { position:absolute; height:4px; border-radius:2px; top:2px; }
.mini-bar:nth-child(2) { top:6px; }
.mini-bar:nth-child(3) { top:10px; }

.words table { font-size:.8rem; }
.words td { white-space:normal; }
.words th[scope="row"] {
  font-family:"IBM Plex Mono", ui-monospace, monospace; font-size:.79rem;
  font-weight:500; color:var(--ink); text-align:left; padding:7px 13px;
  border-bottom:1px solid var(--rule); position:static; background:none;
  text-transform:none; letter-spacing:0; white-space:nowrap;
}
.words .term { color:var(--ink); }
.words .z { color:var(--ink-3); font-size:.72rem; margin-left:3px;
            font-variant-numeric:tabular-nums; }

nav.index { display:flex; flex-wrap:wrap; gap:8px 20px; font-size:.85rem;
            margin:0 0 8px; }
nav.index a { font-weight:500; text-decoration:none; color:var(--ink-2); }
nav.index a:hover { color:var(--accent-ink); }

code, .mono {
  font-family:"IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size:.86em; background:var(--well); padding:1px 5px; border-radius:4px;
  color:var(--ink-2);
}
.filelist { list-style:none; padding:0; margin:12px 0 0;
            font-family:"IBM Plex Mono", ui-monospace, monospace; font-size:.8rem; }
.filelist li { padding:4px 0; border-bottom:1px solid var(--rule);
               color:var(--ink-3); display:flex; gap:14px; flex-wrap:wrap; }
.filelist b { color:var(--ink-2); font-weight:500; flex:none; min-width:min(320px,100%); }

footer.colophon { border-top:1px solid var(--rule); margin-top:60px; padding-top:22px;
                  color:var(--ink-3); font-size:.84rem; }
@media (prefers-reduced-motion: reduce) {
  * { transition:none !important; animation:none !important; }
}
@media (max-width:640px) {
  body { font-size:16px; }
  .wrap { padding:0 18px 64px; }
}
"""

FONT_LINK = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
             '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
             '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
             'family=IBM+Plex+Mono:wght@400;500&'
             'family=IBM+Plex+Sans:wght@400;500;600&'
             'family=IBM+Plex+Serif:ital,wght@0,400;0,600;1,400&display=swap">')


def document(title: str, body: str) -> str:
    """The page, ready for the artifact wrapper: no doctype, html, head, body."""
    return (f"<title>{_esc(title)}</title>\n{FONT_LINK}\n"
            f"<style>{STYLE}</style>\n{body}\n")


def standalone(title: str, body: str) -> str:
    """The same page as a file that opens on its own from disk."""
    return ("<!doctype html>\n<html lang=\"en\">\n<head>\n"
            "<meta charset=\"utf-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">\n"
            f"<title>{_esc(title)}</title>\n{FONT_LINK}\n"
            f"<style>{STYLE}</style>\n</head>\n<body>\n{body}\n</body>\n</html>\n")
