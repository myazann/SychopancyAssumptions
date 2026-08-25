#!/usr/bin/env python3
"""First-pass descriptives on a parsed assumptions run.

    python scripts/summarize_assumptions.py results/gemma3-12b_openended_assumptions.parquet

Four tables, each aimed at one of the questions the design was built for:

  1. instrument health -- parse rate and reply rate per persona facet. Read this
     first. A facet that parses worse than the others differs in format
     compliance, and that difference will masquerade as a finding in the rest.

  2. what the model assumes, by facet -- the most frequent top-1 assumption
     per facet, with lift against the persona-free control. Lift > 1 means
     disclosing that facet makes the model reach for that read of the user more
     often than it does with no persona at all.

  3. confidence -- top-1 probability mass and entropy over the k models, by
     facet and framing. A persona that makes the model *more certain* who it is
     talking to is doing something different from one that shifts *which* read
     it settles on, and the two are worth separating.

  4. framing sensitivity -- for the same person and the same dilemma told from
     either side, does the top-1 assumption survive the flip? Low agreement
     means the model's read of the user tracks the story it was told rather
     than the person it was told about, which is the sycophancy-shaped result.

These are descriptives, not tests. Labels are free text grouped by normalized
string, so near-synonyms ("wants validation" / "seeking validation") land in
separate rows; clustering or embedding them is the next step, and a deliberate
one -- how the labels are grouped decides what the frequency tables say.
"""
from __future__ import annotations

import argparse
import math
import os
import pathlib
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from syco.data import NO_PERSONA

# A cell's identity in the lean results table, which carries no cell_key.
CELL_KEYS = ("persona_type", "persona_id", "prompt_type", "prompt_id", "rep")

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


def entropy(probs) -> float:
    probs = [p for p in probs if p and p > 0]
    return -sum(p * math.log(p, 2) for p in probs) if probs else float("nan")


# Every format parse_assumptions.py can write, read back by extension.
READERS = {
    ".parquet": pd.read_parquet,
    ".csv": pd.read_csv,
    ".json": pd.read_json,
    ".jsonl": lambda p: pd.read_json(p, lines=True),
}


def load(path) -> pd.DataFrame:
    suffix = pathlib.Path(path).suffix.lower()
    reader = READERS.get(suffix)
    if reader is None:
        raise SystemExit(f"Cannot read {suffix or path!r}. "
                         f"Expected one of: {', '.join(sorted(READERS))}")
    df = reader(path)
    if "assumption" not in df.columns:
        raise SystemExit(
            f"{path} has no `assumption` column -- is it a *_cells file, or a "
            "table from before the model_name -> assumption rename? Re-run "
            "parse_assumptions.py on the JSONL to regenerate it."
        )
    df["label"] = df["assumption"].map(normalize_label)
    return df


def health(df: pd.DataFrame) -> pd.DataFrame:
    # The lean results table has no cell_key, so a cell is identified by its
    # design coordinates -- which is what makes a cell unique anyway.
    keys = [c for c in ("cell_key",) if c in df.columns] or CELL_KEYS
    cells = df.drop_duplicates(keys)
    agg = {
        "cells": (keys[0], "size"),
        "parsed": ("parse_status", lambda s: (s != "failed").mean()),
        "clean": ("parse_status", lambda s: (s == "clean").mean()),
    }
    if "has_response" in cells.columns:
        agg["reply"] = ("has_response", "mean")
    return cells.groupby("persona_type").agg(**agg).sort_values("parsed")


def top_labels(df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """Most frequent top-1 assumption per facet, with lift vs. the control."""
    top1 = df[(df["rank"] == 0) & df["label"].astype(bool)]
    if top1.empty:
        return pd.DataFrame()

    control = top1[top1.persona_type == NO_PERSONA]
    base = control.label.value_counts(normalize=True) if len(control) \
        else top1.label.value_counts(normalize=True)

    rows = []
    for facet, sub in top1.groupby("persona_type"):
        share = sub.label.value_counts(normalize=True)
        for label, frac in share.head(n).items():
            b = base.get(label, float("nan"))
            rows.append({
                "persona_type": facet, "label": label,
                "share": frac, "n": int((sub.label == label).sum()),
                "control_share": b,
                "lift": (frac / b) if b and b == b and b > 0 else float("nan"),
            })
    return pd.DataFrame(rows)


def confidence(df: pd.DataFrame) -> pd.DataFrame:
    keys = [c for c in CELL_KEYS if c in df.columns]
    per_cell = df.groupby(keys, as_index=False).agg(
        top1=("probability_norm", "max"),
        ent=("probability_norm", lambda s: entropy(list(s))),
    )
    return per_cell.groupby(["persona_type", "prompt_type"]).agg(
        cells=(keys[0], "size"), top1_mass=("top1", "mean"),
        entropy_bits=("ent", "mean"),
    ).round(3)


def framing_flip(df: pd.DataFrame) -> pd.DataFrame:
    """Does the top-1 assumption survive retelling the dilemma from the other
    side? One row per facet: the share of (person, dilemma) pairs that keep it."""
    top1 = df[(df["rank"] == 0) & df["label"].astype(bool)]
    index = [c for c in ("persona_type", "persona_id", "prompt_id", "rep")
             if c in top1.columns]
    wide = top1.pivot_table(index=index, columns="prompt_type",
                            values="label", aggfunc="first")
    if not {"original_post", "flipped_story"}.issubset(wide.columns):
        return pd.DataFrame()
    both = wide.dropna(subset=["original_post", "flipped_story"])
    if both.empty:
        return pd.DataFrame()
    both = both.assign(same=both.original_post == both.flipped_story)
    return (both.reset_index().groupby("persona_type")
            .agg(pairs=("same", "size"), kept_top1=("same", "mean")).round(3)
            .sort_values("kept_top1"))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="*_assumptions.parquet from parse_assumptions.py")
    p.add_argument("--top", type=int, default=3, help="labels per facet to print")
    p.add_argument("--out", default=None, help="also write the tables to an .xlsx")
    args = p.parse_args(argv)

    df = load(args.input)
    tables = {
        "health": health(df),
        "top_labels": top_labels(df, args.top),
        "confidence": confidence(df),
        "framing_flip": framing_flip(df),
    }
    titles = {
        "health": "1. instrument health (sorted worst-parsing first)",
        "top_labels": f"2. most frequent top-1 assumption per facet (top {args.top}), "
                      "lift vs. the persona-free control",
        "confidence": "3. confidence: top-1 probability mass and entropy over the k models",
        "framing_flip": "4. framing sensitivity: share of (person, dilemma) pairs whose "
                        "top-1 assumption survives the flip",
    }
    for name, table in tables.items():
        print(f"\n{titles[name]}\n{'-' * len(titles[name])}")
        print("(nothing to show)" if table.empty else table.to_string())

    if args.out:
        with pd.ExcelWriter(args.out) as writer:
            for name, table in tables.items():
                if not table.empty:
                    table.to_excel(writer, sheet_name=name[:31])
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
