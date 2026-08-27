#!/usr/bin/env python3
"""First-pass descriptives on a parsed verbalized-assumptions run.

    python -m syco summarize results/gemma3-12b_openended_assumptions.parquet

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

For a `*_structured.parquet` input, the command instead reports instrument
health, mean 0-1 score per fixed dimension and condition (including the delta
from the persona-free control), and sensitivity to flipping the dilemma.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from syco.data import NO_PERSONA
from syco.tables import (  # noqa: F401  -- re-exported for existing importers
    CELL_KEYS,
    READERS,
    cell_keys,
    load,
    model_dimensions,
    normalize_label,
)


def entropy(probs) -> float:
    probs = [p for p in probs if p and p > 0]
    return -sum(p * math.log(p, 2) for p in probs) if probs else float("nan")


def health(df: pd.DataFrame) -> pd.DataFrame:
    # The lean results table has no cell_key, so a cell is identified by its
    # design coordinates -- which is what makes a cell unique anyway.
    keys = cell_keys(df)
    cells = df.drop_duplicates(keys)
    agg = {
        "cells": (keys[0], "size"),
        "parsed": (
            "parse_status",
            lambda status: status.isin(("clean", "repaired", "salvaged")).mean(),
        ),
        "clean": ("parse_status", lambda s: (s == "clean").mean()),
    }
    if "has_response" in cells.columns:
        agg["reply"] = ("has_response", "mean")
    groups = [*model_dimensions(cells), "persona_type"]
    return cells.groupby(groups).agg(**agg).sort_values("parsed")


def top_labels(df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """Most frequent top-1 assumption per facet, with lift vs. the control."""
    top1 = df[(df["rank"] == 0) & df["label"].astype(bool)]
    if top1.empty:
        return pd.DataFrame()

    rows = []
    dimensions = model_dimensions(top1)
    outer = top1.groupby(dimensions, dropna=False) if dimensions else [((), top1)]
    for group_value, model_df in outer:
        values = group_value if isinstance(group_value, tuple) else (group_value,)
        identity = dict(zip(dimensions, values))
        control = model_df[model_df.persona_type == NO_PERSONA]
        base = control.label.value_counts(normalize=True)
        for facet, sub in model_df.groupby("persona_type"):
            share = sub.label.value_counts(normalize=True)
            for label, frac in share.head(n).items():
                b = base.get(label, float("nan"))
                rows.append({
                    **identity,
                    "persona_type": facet, "label": label,
                    "share": frac, "n": int((sub.label == label).sum()),
                    "control_share": b,
                    "lift": (frac / b) if pd.notna(b) and b > 0 else float("nan"),
                })
    return pd.DataFrame(rows)


def confidence(df: pd.DataFrame) -> pd.DataFrame:
    keys = [c for c in CELL_KEYS if c in df.columns]
    per_cell = df.groupby(keys, as_index=False).agg(
        top1=("probability_norm", "max"),
        ent=("probability_norm", lambda s: entropy(list(s))),
    )
    groups = [*model_dimensions(per_cell), "persona_type", "prompt_type"]
    return per_cell.groupby(groups).agg(
        cells=(keys[0], "size"), top1_mass=("top1", "mean"),
        entropy_bits=("ent", "mean"),
    ).round(3)


def framing_flip(df: pd.DataFrame) -> pd.DataFrame:
    """Does the top-1 assumption survive retelling the dilemma from the other
    side? One row per facet: the share of (person, dilemma) pairs that keep it."""
    top1 = df[(df["rank"] == 0) & df["label"].astype(bool)]
    index = [
        c for c in (
            "run_id", "probe", "persona_type",
            "persona_id", "prompt_id", "rep",
        )
             if c in top1.columns]
    wide = top1.pivot_table(index=index, columns="prompt_type",
                            values="label", aggfunc="first")
    if not {"original_post", "flipped_story"}.issubset(wide.columns):
        return pd.DataFrame()
    both = wide.dropna(subset=["original_post", "flipped_story"])
    if both.empty:
        return pd.DataFrame()
    both = both.assign(same=both.original_post == both.flipped_story)
    groups = [*model_dimensions(both.reset_index()), "persona_type"]
    return (both.reset_index().groupby(groups)
            .agg(pairs=("same", "size"), kept_top1=("same", "mean")).round(3)
            .sort_values("kept_top1"))


def structured_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Mean fixed-dimension scores, with a condition-matched control contrast."""
    groups = [
        *model_dimensions(df), "persona_type", "prompt_type", "dimension",
    ]
    table = (df.groupby(groups, dropna=False, observed=True)
             .agg(cells=("score", "size"), scored=("score", "count"),
                  mean_score=("score", "mean"), std_score=("score", "std"))
             .reset_index())
    identity = [*model_dimensions(table), "prompt_type", "dimension"]
    control = (table[table.persona_type == NO_PERSONA]
               [identity + ["mean_score"]]
               .rename(columns={"mean_score": "control_mean"}))
    if control.empty:
        table["control_mean"] = float("nan")
    else:
        table = table.merge(control, on=identity, how="left", validate="many_to_one")
    table["delta_vs_control"] = table.mean_score - table.control_mean
    table["ratio_vs_control"] = table.mean_score / table.control_mean.where(
        table.control_mean > 0
    )
    return table.sort_values([*model_dimensions(table), "dimension",
                              "prompt_type", "persona_type"])


def structured_framing_flip(df: pd.DataFrame) -> pd.DataFrame:
    """How much each structured score moves when the same dilemma is flipped."""
    scored = df.dropna(subset=["score"])
    index = [
        column for column in (
            "run_id", "probe", "persona_type", "persona_id",
            "prompt_id", "rep", "dimension",
        ) if column in scored.columns
    ]
    wide = scored.pivot_table(
        index=index, columns="prompt_type", values="score", aggfunc="first"
    )
    framings = {"original_post", "flipped_story"}
    if not framings.issubset(wide.columns):
        return pd.DataFrame()
    both = wide.dropna(subset=sorted(framings)).reset_index()
    if both.empty:
        return pd.DataFrame()
    both["signed_change"] = both.flipped_story - both.original_post
    both["absolute_change"] = both.signed_change.abs()
    groups = [*model_dimensions(both), "persona_type", "dimension"]
    rows = []
    for values, subset in both.groupby(groups, dropna=False, observed=True):
        values = values if isinstance(values, tuple) else (values,)
        correlation = float("nan")
        if (len(subset) > 1 and subset.original_post.nunique() > 1
                and subset.flipped_story.nunique() > 1):
            correlation = subset.original_post.corr(subset.flipped_story)
        rows.append({
            **dict(zip(groups, values)),
            "pairs": len(subset),
            "mean_absolute_change": subset.absolute_change.mean(),
            "mean_signed_change": subset.signed_change.mean(),
            "score_correlation": correlation,
        })
    return pd.DataFrame(rows).sort_values(
        [*model_dimensions(both), "dimension", "persona_type"]
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="*_assumptions or *_structured table from parse")
    p.add_argument("--top", type=int, default=3, help="labels per facet to print")
    p.add_argument("--out", default=None, help="also write the tables to an .xlsx")
    args = p.parse_args(argv)

    df = load(args.input)
    if {"dimension", "score"}.issubset(df.columns):
        tables = {
            "health": health(df),
            "scores": structured_scores(df),
            "framing_flip": structured_framing_flip(df),
        }
        titles = {
            "health": "1. instrument health (sorted worst-parsing first)",
            "scores": "2. structured scores by dimension, facet, and framing",
            "framing_flip": "3. framing sensitivity of structured scores",
        }
    else:
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
