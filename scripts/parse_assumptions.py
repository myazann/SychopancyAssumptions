#!/usr/bin/env python3
"""Turn a raw assumptions run into one tidy results table.

    python -m syco parse results/gemma3-12b_openended.jsonl

writes, next to the input, `*_assumptions.parquet` -- one row per verbalized
assumption:

    persona_type  persona_id  prompt_type  prompt_id  rep
    rank  assumption  description  probability  probability_norm  parse_status

k rows per cell at `--n-models k`. `--full` adds per-cell parse diagnostics;
`--cells` additionally writes the one-row-per-cell table. Constant model
provenance remains in the adjacent run manifest.

Parsing is separate from the run on purpose: the JSONL keeps every completion
verbatim, so a parser fix is a re-parse, not a re-run. That is also why this
step drops columns freely -- nothing is lost, it is all still in the JSONL.

The header it prints is the first thing to read after a run. `clean` /
`repaired` / `salvaged` / `failed` is the instrument's own health check -- if a
persona facet parses noticeably worse than the others, differences between
facets are partly differences in how well the model followed the format, and
that has to be handled before any of it is read as a finding.

Structured runs use the same command and write `*_structured.parquet`, with
one row per fixed dimension (`dimension`, `score`, `explanation`).
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from syco.parse import (
    parse_completion,
    parse_structured,
    to_records,
    to_structured_records,
)
from syco.prompts import STRUCTURED_DIMENSIONS
from syco.store import canonical_rows, read_rows


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="results JSONL from run_assumptions.py")
    p.add_argument("--out-prefix", default=None,
                   help="output prefix (default: the input path without .jsonl)")
    p.add_argument("--format", choices=("parquet", "csv", "json", "jsonl"),
                   default="parquet",
                   help="parquet (default: typed, compact, needs pyarrow) | csv | "
                        "json (one array, readable) | jsonl (one object per line, "
                        "streamable and diffable)")
    p.add_argument("--full", action="store_true",
                   help="keep per-cell parsing diagnostics, not just the lean "
                        "results set (constant model provenance stays in the manifest)")
    p.add_argument("--cells", action="store_true",
                   help="also write *_cells.<fmt>: one row per cell rather than "
                        "per assumption, for run-quality checks")
    p.add_argument("--keep-response", action="store_true",
                   help="store the model's reply in the cells table (implies "
                        "--cells). Large -- the reply is already in the JSONL")
    return p.parse_args(argv)



def _parse_structured_file(args, rows, kind, attempt_counts) -> int:
    """Parse a structured probe (`4dims`, `supporttypes`) into one row per
    scored dimension. Kept separate from the open-ended path because the two
    have different schemas and different invariants -- the structured
    dimensions are independent and are deliberately not normalized."""
    records, cell_records = [], []
    status_counts, by_facet = Counter(), {}
    error_count = 0

    for row in rows:
        if row.get("error"):
            error_count += 1
        parsed = parse_structured(row.get("raw", ""), kind)
        records.extend(to_structured_records(row, parsed, full=args.full))
        status_counts[parsed.status] += 1
        by_facet.setdefault(row.get("persona_type"), Counter())[parsed.status] += 1
        cell = {k: row.get(k) for k in (
            "cell_key", "run_id", "persona_type", "persona_id", "prompt_type",
            "prompt_id", "rep", "probe", "persona_turns", "persona_recovered",
            "model_ref", "quantization", "temperature", "prompt_digest",
            "error", "timestamp",
        )}
        cell["attempt_count"] = attempt_counts.get(row.get("cell_key"), 1)
        cell["n_dimensions_asked"] = row.get("n_dimensions_asked")
        cell.update(parse_status=parsed.status, n_dimensions=parsed.n_dimensions,
                    n_scored=parsed.n_scored,
                    has_response=parsed.has_response,
                    response_chars=len(parsed.response),
                    raw_chars=len(row.get("raw", "")), parse_notes=parsed.notes)
        if args.keep_response:
            cell["response"] = parsed.response
        cell_records.append(cell)

    beliefs = pd.DataFrame(records)
    cells = pd.DataFrame(cell_records)
    default_prefix = args.input.removesuffix(".jsonl")
    prefix = args.out_prefix or default_prefix
    writers = {
        "parquet": lambda df, path: df.to_parquet(path, index=False),
        "csv": lambda df, path: df.to_csv(path, index=False),
        "json": lambda df, path: df.to_json(path, orient="records", indent=2,
                                            force_ascii=False),
        "jsonl": lambda df, path: df.to_json(path, orient="records", lines=True,
                                             force_ascii=False),
    }
    write = writers[args.format]
    out = f"{prefix}_structured.{args.format}"
    write(beliefs, out)

    n_cells = len(cells)
    total = sum(status_counts.values()) or 1
    scored_count = int(beliefs["score"].notna().sum()) if "score" in beliefs else 0
    print(
        f"{n_cells} cell(s) -> {len(beliefs)} dimension row(s), "
        f"{scored_count} scored  [probe: {kind}]"
    )
    if attempt_counts and sum(attempt_counts.values()) > n_cells:
        print(
            f"attempt history: {sum(attempt_counts.values())} row(s), "
            f"{sum(count > 1 for count in attempt_counts.values())} retried cell(s), "
            f"{sum(max(0, count - 1) for count in attempt_counts.values())} "
            "superseded attempt(s)"
        )
    print("parse status: " + ", ".join(
        f"{name}={count} ({count / total:.1%})"
        for name, count in status_counts.most_common()))
    if error_count:
        print(f"unresolved errors: {error_count} ({error_count / max(n_cells, 1):.1%})")
    scored = beliefs.dropna(subset=["score"]) if "score" in beliefs else beliefs
    if len(scored):
        print("\nmean score per dimension")
        print(scored.groupby("dimension")["score"].agg(["count", "mean", "std"])
              .to_string(float_format=lambda v: f"{v:.3f}"))
    worst = sorted(
        (
            (
                facet,
                (counts.get("failed", 0) + counts.get("invalid_order", 0))
                / max(sum(counts.values()), 1),
            )
            for facet, counts in by_facet.items()
        ),
        key=lambda item: -item[1],
    )
    if worst and worst[0][1] > 0:
        print("failure rate by persona facet (highest first): " + ", ".join(
            f"{facet}={rate:.1%}" for facet, rate in worst[:5] if rate > 0
        ))
    print(f"\nwrote {out}")
    if args.cells or args.keep_response:
        cells_out = f"{prefix}_cells.{args.format}"
        write(cells, cells_out)
        print(f"      {cells_out}")
    return 1 if error_count else 0


def main(argv=None) -> int:
    args = parse_args(argv)
    attempts = read_rows(args.input)
    if not attempts:
        print(f"no rows in {args.input}")
        return 1
    rows, retry_diagnostics = canonical_rows(attempts)
    attempt_counts = Counter(row.get("cell_key") for row in attempts)

    # Which probe produced this file decides which table it parses into. The
    # probe label is stamped on every row by the runner, so this is read from
    # the data rather than passed in and possibly mismatched.
    labels = {str(r.get("probe") or "") for r in rows}
    nonempty_labels = labels - {""}
    if len(nonempty_labels) > 1:
        print(
            "mixed probe labels in one raw file: " + ", ".join(sorted(nonempty_labels)),
            file=sys.stderr,
        )
        return 2
    label = next(iter(nonempty_labels), "")
    if label in STRUCTURED_DIMENSIONS:
        return _parse_structured_file(args, rows, label, attempt_counts)

    assumption_records, cell_records = [], []
    status_counts, by_facet = Counter(), {}
    error_count = 0

    for row in rows:
        if row.get("error"):
            error_count += 1
        asked = row.get("n_assumptions_asked", row.get("n_models_asked"))
        parsed = parse_completion(row.get("raw", ""), expected_n=asked)
        assumption_records.extend(to_records(row, parsed, full=args.full))
        status_counts[parsed.status] += 1
        by_facet.setdefault(row.get("persona_type"), Counter())[parsed.status] += 1

        cell = {k: row.get(k) for k in (
            "cell_key", "run_id", "persona_type", "persona_id", "prompt_type", "prompt_id",
            "rep", "probe", "persona_turns",
            "persona_recovered", "model_ref", "quantization",
            "temperature", "thinking_applied", "thinking_standardized",
            "prompt_digest", "error", "timestamp",
        )}
        cell["n_assumptions_asked"] = row.get("n_assumptions_asked",
                                              row.get("n_models_asked"))
        cell["attempt_count"] = attempt_counts.get(row.get("cell_key"), 1)
        cell.update(parse_status=parsed.status, n_assumptions=parsed.n_assumptions,
                    prob_sum=parsed.prob_sum, has_response=parsed.has_response,
                    response_chars=len(parsed.response),
                    raw_chars=len(row.get("raw", "")), parse_notes=parsed.notes)
        if args.keep_response:
            cell["response"] = parsed.response
        cell_records.append(cell)

    assumptions = pd.DataFrame(assumption_records)
    cells = pd.DataFrame(cell_records)

    default_prefix = args.input.removesuffix(".jsonl")
    prefix = args.out_prefix or default_prefix
    # Parquet is the default because these tables get wide and long -- it keeps
    # dtypes (so `rank` stays an int and a missing probability stays null rather
    # than becoming the string "nan") and reads back in one line. None of that is
    # required: the raw run output is already JSONL, and every downstream step
    # here takes a DataFrame, so pick whichever you would rather open.
    writers = {
        "parquet": lambda df, path: df.to_parquet(path, index=False),
        "csv": lambda df, path: df.to_csv(path, index=False),
        "json": lambda df, path: df.to_json(path, orient="records", indent=2,
                                            force_ascii=False),
        "jsonl": lambda df, path: df.to_json(path, orient="records", lines=True,
                                             force_ascii=False),
    }
    writer = writers[args.format]
    ext = args.format
    a_path = f"{prefix}_assumptions.{ext}"
    writer(assumptions, a_path)
    written = [a_path]
    if args.cells or args.keep_response:
        c_path = f"{prefix}_cells.{ext}"
        writer(cells, c_path)
        written.append(c_path)

    # -- the health check --------------------------------------------------
    total = len(rows)
    print(f"{total} cell(s) -> {len(assumptions)} verbalized assumption(s)")
    if retry_diagnostics["extra_attempts"]:
        print(
            f"attempt history: {retry_diagnostics['attempts']} row(s), "
            f"{retry_diagnostics['retried_cells']} retried cell(s), "
            f"{retry_diagnostics['extra_attempts']} superseded attempt(s)"
        )
    if error_count:
        print(f"unresolved errors: {error_count} ({error_count / total:.1%})")
    print("parse status: " + ", ".join(
        f"{k}={v} ({v / total:.1%})" for k, v in status_counts.most_common()))
    usable = cells[cells.parse_status.isin(("clean", "repaired", "salvaged"))]
    if len(usable):
        asked = cells.n_assumptions_asked.iloc[0]
        print(f"assumptions per cell: mean {usable.n_assumptions.mean():.2f} "
              f"(asked for {asked})")
        sums = usable.prob_sum.dropna()
        if len(sums):
            print(f"probability sum: mean {sums.mean():.3f}, "
                  f"{(sums.sub(1).abs() < 0.01).mean():.1%} within 0.01 of 1.0")
        print(f"reply present:   {usable.has_response.mean():.1%} of parsed cells")

    worst = sorted(((f, (c.get("failed", 0) + c.get("invalid_order", 0)) /
                     max(sum(c.values()), 1))
                    for f, c in by_facet.items()), key=lambda x: -x[1])
    if worst and worst[0][1] > 0:
        print("failure rate by persona facet (highest first): " + ", ".join(
            f"{f}={r:.1%}" for f, r in worst[:5] if r > 0))

    print("\nwrote " + "\n      ".join(written))
    return 1 if error_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
