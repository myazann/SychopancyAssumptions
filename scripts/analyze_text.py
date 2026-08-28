#!/usr/bin/env python3
"""Analyze persona or response text without treating language as sycophancy.

Examples:

    # Lexical features on long-form model responses
    python -m syco text features files/gemma-3-12b-it_long_results.pkl \
        --text-column model_answer --method stance --out response_features.parquet

    # LIWC features computed through pyliwc and an activated LIWC installation
    python -m syco text features files/gemma-3-12b-it_long_results.pkl \
        --text-column model_answer --method liwc --liwc-cli LIWC-22-cli \
        --out response_liwc.parquet

    # Words in persona self-descriptions associated with extracted assumptions
    python -m syco text words files/base_data_persona.gz \
        results/Gemma3-12B/openended3_assumptions.parquet \
        --text-column persona_text --persona-role user --min-count 5 \
        --out persona_assumption_words.parquet

The feature methods are descriptive.  They are valid for either source of
text and are never converted into an endorsement or sycophancy score.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from syco import text_analysis as text
from syco.tables import load as load_assumptions


def _load(path) -> pd.DataFrame:
    suffix = pathlib.Path(path).suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix in (".pkl", ".pickle", ".gz"):
        return pd.read_pickle(path)
    raise SystemExit(
        f"cannot read {path}: expected pickle/gz, parquet, csv, json, or jsonl"
    )


def _write(table: pd.DataFrame, path: str) -> None:
    suffix = pathlib.Path(path).suffix.lower()
    if suffix == ".parquet":
        table.to_parquet(path, index=False)
    elif suffix == ".csv":
        table.to_csv(path, index=False)
    elif suffix == ".jsonl":
        table.to_json(path, orient="records", lines=True)
    else:
        raise SystemExit(f"cannot write {path}: expected .parquet, .csv, or .jsonl")
    print(f"wrote {path} ({len(table):,} rows)")


def _text_column(table: pd.DataFrame, requested: str | None) -> str:
    if requested:
        if requested not in table.columns:
            raise SystemExit(f"input has no {requested!r} column")
        return requested
    for candidate in ("persona_text", "model_answer", "response", "description",
                      "assumption", "text"):
        if candidate in table.columns:
            return candidate
    raise SystemExit("could not infer a text column; pass --text-column")


def _prepare(table: pd.DataFrame, column: str, persona_role: str) -> tuple[pd.DataFrame, str]:
    if column != "persona_text":
        return table, column
    out = table.copy()
    output = f"persona_{persona_role}_text"
    out[output] = text.persona_texts(out, source=column, role=persona_role)
    return out, output


def _features(args) -> int:
    table = _load(args.input)
    column = _text_column(table, args.text_column)
    table, column = _prepare(table, column, args.persona_role)
    methods = args.method or ["stance"]
    analyzed = text.attach_text_features(
        table, column, methods=methods, batch_size=args.batch_size,
        device=args.device, liwc_cli_path=args.liwc_cli,
        liwc_dict=args.liwc_dictionary, liwc_threads=args.liwc_threads,
        liwc_verbose=args.liwc_verbose,
    )
    feature_columns = [column for column in analyzed.columns
                       if column not in table.columns]
    print(f"analyzed {len(analyzed):,} texts from {column!r} with "
          f"{', '.join(methods)}; added {len(feature_columns)} feature columns")
    if args.out:
        _write(analyzed, args.out)
    elif feature_columns:
        print(analyzed[feature_columns].describe().transpose().to_string())
    return 0


def _words(args) -> int:
    if not args.assumptions:
        raise SystemExit("text words needs an assumptions table")
    table = _load(args.input)
    column = _text_column(table, args.text_column)
    table, column = _prepare(table, column, args.persona_role)
    assumptions = load_assumptions(args.assumptions)
    words = text.marked_words_by_assumption(
        table, assumptions, text_column=column, field=args.field,
        keys=args.key, min_count=args.min_count, threshold=args.threshold,
        top=args.top_per_assumption,
    )
    print(f"found {len(words):,} marked word/assumption associations across "
          f"{words[args.field].nunique() if len(words) else 0:,} assumptions")
    if args.out:
        _write(words, args.out)
    else:
        print("(nothing to show)" if words.empty
              else words.head(args.top).to_string(index=False))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m syco text", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("stage", choices=("features", "words"))
    parser.add_argument("input", help="table containing persona or response text")
    parser.add_argument("assumptions", nargs="?", help="words: parsed assumptions table")
    parser.add_argument("--text-column", default=None,
                        help="text column (default: infer a known text column)")
    parser.add_argument("--persona-role", choices=("user", "assistant", "all"),
                        default="user", help="turns to retain when persona_text is a transcript")
    parser.add_argument("--method", action="append", choices=text.TEXT_METHODS,
                        help="features: repeatable analysis method (default: stance)")
    parser.add_argument("--liwc-cli", default="LIWC-22-cli",
                        help="features: LIWC CLI executable used by pyliwc")
    parser.add_argument("--liwc-dictionary", default="LIWC22",
                        help="features: built-in dictionary name or .dicx path")
    parser.add_argument("--liwc-threads", type=int, default=None,
                        help="features: LIWC workers (default: CPU count minus one)")
    parser.add_argument("--liwc-verbose", action="store_true",
                        help="features: show LIWC CLI progress")
    parser.add_argument("--key", action="append", default=None,
                        help="words: repeatable text/assumption join key")
    parser.add_argument("--field", default="label",
                        help="words: assumption or topic column (default: label)")
    parser.add_argument("--min-count", type=int, default=5,
                        help="words: minimum associated text units per assumption")
    parser.add_argument("--threshold", type=float, default=text.MARKED_WORDS_Z,
                        help="words: minimum Fightin' Words z score")
    parser.add_argument("--top-per-assumption", type=int, default=None,
                        help="words: retain at most this many words per assumption")
    parser.add_argument("--top", type=int, default=30,
                        help="rows to print when --out is omitted")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="sentiment/emotion classifier batch size")
    parser.add_argument("--device", default=None,
                        help="sentiment/emotion torch device (default: auto)")
    parser.add_argument("--out", default=None,
                        help="write results to .parquet, .csv, or .jsonl")
    args = parser.parse_args(argv)
    if args.stage == "features" and args.assumptions:
        parser.error("features does not take an assumptions table")
    if args.liwc_threads is not None and args.liwc_threads <= 0:
        parser.error("--liwc-threads must be positive")
    return {"features": _features, "words": _words}[args.stage](args)


if __name__ == "__main__":
    raise SystemExit(main())
