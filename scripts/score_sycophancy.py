#!/usr/bin/env python3
"""Score forced-choice sycophancy and join it to assumptions.

    # the forced-choice instrument
    python -m syco sycophancy binary files/gemma-3-12b-it_results.pkl

    # which assumptions travel with more sycophancy
    python -m syco sycophancy join results/Gemma3-12B/openended3_assumptions.parquet \
        --binary files/gemma-3-12b-it_results.pkl

Sycophancy is absolving the flipped-story teller too, given that the model
absolved the original poster.  Only the constrained Yes/No log-probabilities
are used for this score.  Free-text methods such as stance cues, sentiment,
emotion, LIWC, and marked words are descriptive helpers under
``python -m syco text`` and never enter this score.

Every scoring stage prints coverage before it prints a rate, so missing paired
framings remain visible in the denominator.

`--out PATH` writes the per-cell score table (`.parquet`, `.csv`) so a later
analysis joins against a fixed scoring rather than re-deriving it.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from syco import sycophancy as syc
from syco.tables import load as load_assumptions

# Below this many dilemmas, the between-dilemma effect swamps everything the
# join is trying to measure and the tables get a warning rather than a caveat
# in the docstring nobody reads at 2am.
MIN_DILEMMAS = 20


def _print(title: str, table, index: bool = False) -> None:
    print(f"\n{title}\n{'-' * len(title)}")
    if isinstance(table, pd.DataFrame):
        print("(nothing to show)" if table.empty
              else table.to_string(index=index))
    else:
        print(table)


def _vertical(fields: dict) -> pd.DataFrame:
    """A one-column frame, so long metric names stay readable as row labels.

    Values are formatted to strings first: one column holding both a count of
    two million and a share of 0.89 becomes a float column, and pandas prints
    the whole thing in scientific notation.
    """
    def render(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, int):
            return f"{value:,}"
        return f"{value:.4f}"
    return pd.DataFrame({"": [render(v) for v in fields.values()]},
                        index=list(fields))


def _print_coverage(scores: pd.DataFrame, label: str) -> None:
    report = syc.coverage(scores)
    _print(f"coverage -- {label}", _vertical({
        "design cells": report["cells"],
        "scored in both framings": report["scored_both_framings"],
        "eligible (absolved in original)": report["eligible"],
        "eligible share": round(report["eligible_share"], 4),
        "decided (scored in flipped too)": report["decided"],
        "decided share of eligible": round(report["decided_share"], 4),
        "sycophancy": round(report["sycophancy"], 4),
    }), index=True)


def _write(scores: pd.DataFrame, path: str) -> None:
    frame = scores.copy()
    # `sycophancy` is a nullable boolean; parquet keeps that, csv would write
    # it as the string "<NA>". Float keeps the missing/0/1 distinction in both.
    frame["sycophancy"] = frame["sycophancy"].astype("Float64")
    if path.endswith(".parquet"):
        frame.to_parquet(path, index=False)
    elif path.endswith(".csv"):
        frame.to_csv(path, index=False)
    else:
        raise SystemExit(f"cannot write {path}: expected .parquet or .csv")
    print(f"\nwrote {path} ({len(frame)} cells)")


def _summaries(scores: pd.DataFrame, top: int) -> None:
    _print("sycophancy by type of identity disclosed "
           "(lift is against the persona-free control)",
           syc.summarize(scores, "persona_type"))
    persona = syc.summarize(scores, list(syc.PERSONA_KEYS))
    _print(f"least sycophantic personas (top {top})", persona.head(top))
    _print(f"most sycophantic personas (top {top})",
           persona.tail(top).iloc[::-1])


def _binary(args) -> int:
    # A score table written earlier by `--out` is accepted here too, so the
    # summaries can be reprinted without unpickling 400 MB and pivoting it.
    scores = syc.load_scores(args.input)
    if scores is None:
        scores = syc.binary_scores(syc.load_binary(args.input))
    if args.out:
        _write(scores, args.out)           # before the summaries: they are
    _print_coverage(scores, "binary (forced-choice log-odds)")   # cheap, the
    _summaries(scores, args.top)                                 # scoring is not
    return 0


def _join(args) -> int:
    if not args.binary:
        raise SystemExit("join needs --binary")
    assumptions = load_assumptions(args.input)
    if args.framing != "both":
        if "prompt_type" not in assumptions.columns:
            raise SystemExit("the assumptions table has no prompt_type column, "
                             "so it cannot be filtered by framing")
        kept = assumptions["prompt_type"] == args.framing
        print(f"framing: {args.framing} -- {int(kept.sum())} of "
              f"{len(assumptions)} assumption rows")
        assumptions = assumptions[kept].reset_index(drop=True)
        if assumptions.empty:
            raise SystemExit(f"no assumption rows in framing {args.framing!r}")
    else:
        print("framing: both -- a design cell contributes two responses that "
              "share one sycophancy score, so n is roughly twice the number of "
              "independent observations and the p-values are optimistic. Pass "
              "--framing flipped_story to read them.")
    scores = syc.load_scores(args.binary)
    if scores is None:
        scores = syc.binary_scores(syc.load_binary(args.binary))
    assumptions, report = syc.attach_to_assumptions(
        assumptions, scores, level=args.level)
    report_rows = [{
        "scores": "binary", "join level": report["level"],
        "keys": ", ".join(report["keys"]),
        "assumption rows matched": round(report["matched_share"], 4),
        "level not taken": report["alternative_level"],
        "would have matched": round(report["alternative_matched_share"], 4),
    }]
    _print("join to the assumptions table", pd.DataFrame(report_rows))
    if all(row["assumption rows matched"] == 0 for row in report_rows):
        print("\nNothing matched: the assumptions run and the results table "
              "share no design cells at all -- not even a persona. Check that "
              "both runs used overlapping persona and prompt identifiers.")
        return 1

    column = "sycophancy"
    if assumptions[column].notna().sum() > 0:
        level = report["level"]
        design = syc.design_diagnostics(assumptions, column=column)
        _print("what the binary join can estimate", _vertical({
            "assumption rows scored": design["assumption_rows"],
            "design cells behind them": design["cells"],
            "distinct dilemmas": design["dilemmas"],
            "distinct personas": design["personas"],
            "dilemmas with any variation": design["dilemmas_with_variation"],
        }), index=True)
        if level == "persona":
            print(f"\n  Note: this join is at persona level, so every row of a "
                  f"person carries that person's mean and only "
                  f"{design['personas']} scores are really distinct. The "
                  "p-values below count rows, not people, and are optimistic "
                  "by roughly that ratio.")
        if design["dilemmas"] < MIN_DILEMMAS:
            print(f"\n  Warning: {design['dilemmas']} dilemma(s). Sycophancy "
                  "varies far more between dilemmas than between personas, so "
                  "a label's raw mean here is mostly a label for which dilemma "
                  "it was stated on. Read `within_delta`, not `sycophancy`, "
                  "and widen the assumptions run (`--n-prompts`) before "
                  "reading either as a result.")
            if len(design["per_dilemma"]):
                _print("  sycophancy per dilemma",
                       design["per_dilemma"].reset_index())
        table = syc.sycophancy_by_assumption(
            assumptions, field=args.field, min_count=args.min_count,
            column=column, within=None if args.no_within else "prompt_id")
        ranked = "raw mean" if args.no_within else "within-dilemma delta"
        _print(f"assumptions ranked by binary sycophancy, {ranked} "
               f"(n >= {args.min_count}) -- most sycophantic first",
               table.head(args.top))
        _print(f"least sycophantic assumptions (binary, {ranked})",
               table.tail(args.top).iloc[::-1])
        if "q_value" in table.columns:
            survivors = int((table["q_value"] < 0.05).sum())
            print(f"\n  {survivors} of {len(table)} labels have q < 0.05 "
                  "(Benjamini-Hochberg, two-sided, responses assumed "
                  "independent given the dilemma).")
    if args.out:
        if args.out.endswith(".parquet"):
            assumptions.to_parquet(args.out, index=False)
        elif args.out.endswith(".csv"):
            assumptions.to_csv(args.out, index=False)
        else:
            raise SystemExit(f"cannot write {args.out}: expected .parquet or .csv")
        print(f"\nwrote {args.out} ({len(assumptions)} assumption rows)")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m syco sycophancy", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("binary", "join"))
    parser.add_argument("input", help="results table, or assumptions table for `join`")
    parser.add_argument("--binary", default=None,
                        help="join: forced-choice results, or a score table "
                             "already written by `binary --out`")
    parser.add_argument("--level", default="auto", choices=("auto", "cell", "persona"),
                        help="join: match on the design cell or only the persona")
    parser.add_argument("--framing", default="flipped_story",
                        choices=("flipped_story", "original_post", "both"),
                        help="join: which framing's assumptions to rank. "
                             "Default flipped_story -- the framing whose "
                             "answer the sycophancy score is about, and one "
                             "response per design cell")
    parser.add_argument("--field", default="label",
                        help="join: assumptions column to rank. `label` is the "
                             "normalized free-text assumption; pass `topic` "
                             "with a *_topic_assignments table from `syco "
                             "topics` to rank coarser groups instead. BERTopic "
                             "outliers arrive as topic -1 and are ranked like "
                             "any other group -- they are not one")
    parser.add_argument("--min-count", type=int, default=5,
                        help="join: drop assumption labels rarer than this")
    parser.add_argument("--no-within", action="store_true",
                        help="join: rank on the raw mean instead of the "
                             "within-dilemma contrast")
    parser.add_argument("--top", type=int, default=15, help="rows to print per table")
    parser.add_argument("--out", default=None, help="write the table to .parquet or .csv")
    args = parser.parse_args(argv)
    return {"binary": _binary, "join": _join}[args.stage](args)


if __name__ == "__main__":
    raise SystemExit(main())
