"""Reading a parsed assumptions table, and the conventions shared across the
analysis steps.

`parse_assumptions.py` writes one row per verbalized assumption. Everything
downstream -- the descriptives in `summarize_assumptions.py`, the content
analysis in `syco.topics` -- reads that same table and has to agree on three
things, so they live here rather than in whichever script needed them first:

* **what a cell is.** The lean results table carries no `cell_key`, so a cell
  is its design coordinates. This is the denominator whenever a quantity is
  "per response" rather than "per assumption".
* **what must never be pooled.** Two runs, probes, or history modes in one frame
  are different experiments. A run ID identifies its model through the adjacent
  manifest without repeating constant model columns on every assumption.
* **how a free-text label is normalized.** Deliberately shallow, and shared so
  that the label grouped in one table is the same label in the next.
"""
from __future__ import annotations

import pathlib
import re

import pandas as pd

# A cell's identity in the lean results table, which carries no cell_key.
CELL_KEYS = (
    "run_id", "probe", "persona_type",
    "persona_id", "prompt_type", "prompt_id", "rep",
)

# Dimensions that must never be pooled implicitly across experiments.
MODEL_DIMENSIONS = ("run_id", "probe")

_STRIP_RE = re.compile(r"[^a-z0-9 ]+")
_LEAD_RE = re.compile(r"^(the\s+)?(user|person|they|he|she)\s+(is|wants|seeks|needs)\s+")


def normalize_label(text) -> str:
    """Group trivially-different labels. Deliberately shallow -- it collapses
    case, punctuation and a leading 'the user is', and nothing else."""
    if not isinstance(text, str):
        return ""
    text = _STRIP_RE.sub(" ", text.lower()).strip()
    text = _LEAD_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def model_dimensions(df: pd.DataFrame) -> list[str]:
    """The experiment-identity columns actually present in `df`."""
    return [column for column in MODEL_DIMENSIONS if column in df.columns]


def cell_keys(df: pd.DataFrame) -> list[str]:
    """The columns that identify one response (one probe completion)."""
    if "cell_key" in df.columns:
        return ["cell_key"]
    return [column for column in CELL_KEYS if column in df.columns]


def cell_id(df: pd.DataFrame) -> pd.Series:
    """One opaque id per response, for counting distinct responses."""
    keys = cell_keys(df)
    if not keys:
        raise ValueError("table carries no cell-identifying columns")
    if len(keys) == 1:
        return df[keys[0]].astype("string")
    joined = df[keys[0]].astype("string")
    for key in keys[1:]:
        joined = joined.str.cat(df[key].astype("string"), sep="|", na_rep="")
    return joined


# Every format parse_assumptions.py can write, read back by extension.
READERS = {
    ".parquet": pd.read_parquet,
    ".csv": pd.read_csv,
    ".json": pd.read_json,
    ".jsonl": lambda p: pd.read_json(p, lines=True),
}


def load(path) -> pd.DataFrame:
    """Load an open-ended or structured parsed table.

    Open-ended tables receive the normalized ``label`` column used by the text
    analyses. Structured tables already have their fixed ``dimension`` key and
    numeric ``score``, so they are returned without inventing a label.
    """
    suffix = pathlib.Path(path).suffix.lower()
    reader = READERS.get(suffix)
    if reader is None:
        raise SystemExit(f"Cannot read {suffix or path!r}. "
                         f"Expected one of: {', '.join(sorted(READERS))}")
    df = reader(path)
    is_openended = "assumption" in df.columns
    is_structured = {"dimension", "score"}.issubset(df.columns)
    if not (is_openended or is_structured):
        raise SystemExit(
            f"{path} is neither an open-ended assumptions table nor a structured "
            "scores table -- is it a *_cells file? Re-run parse on the JSONL "
            "to regenerate it."
        )
    if is_openended:
        df["label"] = df["assumption"].map(normalize_label)
    # Analysis downstream joins term and topic assignments back on position, so
    # the index has to be unique. csv/json round-trips do not guarantee that.
    return df.reset_index(drop=True)
