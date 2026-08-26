"""Append-only JSONL results, and the resume set built from them.

One row per administered cell, written as it arrives. JSONL because a run over
hundreds of thousands of cells will be interrupted -- by a preempted GPU, a rate
limit, or a laptop lid -- and an append-only file loses at most the last line.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Optional

SCHEMA_VERSION = 2


@dataclass
class AssumptionRecord:
    """One cell's raw result. Parsing happens downstream, never here.

    The raw text is stored verbatim -- with the parse fields left to
    `syco.parse` -- so a parser bug is a re-parse rather than a re-run.
    """
    cell_key: str
    # -- design
    persona_type: str
    persona_id: str
    prompt_type: str
    prompt_id: str
    rep: int
    # -- instrument
    probe: str                    # ProbeSpec.label()
    history_mode: str
    n_assumptions_asked: int
    persona_turns: int
    persona_recovered: bool
    # -- compact run/model audit fields. Full model identity and serving
    # provenance live in the adjacent manifest; the redundant descriptive
    # columns are not repeated here.
    run_id: str = ""
    model_ref: str = ""
    quantization: str = ""
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_output_tokens: int = 0
    thinking_applied: str = ""
    thinking_standardized: bool = True
    # -- observation
    raw: str = ""
    prompt_digest: str = ""
    error: str = ""
    timestamp: str = ""
    schema_version: int = SCHEMA_VERSION

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def append(records, path) -> None:
    """Append and fsync, so an interrupted run keeps everything written."""
    path = str(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(r.to_json() + "\n")
        f.flush()
        os.fsync(f.fileno())


def completed_keys(path, retry_errors: bool = True) -> set:
    """Cell keys already satisfied in `path`.

    With `retry_errors=True` (default) rows carrying an `error` do NOT count, so
    a transient failure is re-attempted on the next run rather than being
    permanently skipped. An empty completion from the model is a real
    observation and does count -- that is data about the model, not a fault.

    Tolerates a truncated final line (a run killed mid-write) and rows from an
    older schema, so resume never dies on a partial file.
    """
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue                     # truncated last line
            if retry_errors and row.get("error"):
                continue
            key = row.get("cell_key")
            if key:
                done.add(key)
    return done


def read_rows(path) -> list:
    """Every well-formed row, as dicts. Skips a truncated final line."""
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def canonical_rows(rows: list) -> tuple[list, dict]:
    """Collapse append-only attempts to one observation per cell.

    The latest successful attempt wins. If a cell has never succeeded, its
    latest failed attempt remains visible. The returned diagnostics make retry
    history explicit without letting it inflate downstream cell counts.
    """
    by_key, unkeyed = {}, []
    attempts = {}
    for row in rows:
        key = row.get("cell_key")
        if not key:
            unkeyed.append(row)
            continue
        attempts[key] = attempts.get(key, 0) + 1
        previous = by_key.get(key)
        if previous is None or previous.get("error") or not row.get("error"):
            by_key[key] = row
    canonical = list(by_key.values()) + unkeyed
    diagnostics = {
        "attempts": len(rows),
        "cells": len(canonical),
        "retried_cells": sum(n > 1 for n in attempts.values()),
        "extra_attempts": sum(max(0, n - 1) for n in attempts.values()),
    }
    return canonical, diagnostics
