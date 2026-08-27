import json
from dataclasses import fields

import pandas as pd

from scripts.parse_assumptions import main as parse_main
from syco.parse import (
    CLEAN,
    INVALID_ORDER,
    REPAIRED,
    SALVAGED,
    parse_completion,
    parse_structured,
    to_records,
    to_structured_records,
)
from syco.prompts import ProbeSpec
from syco.store import AssumptionRecord, canonical_rows

BLOCK = (
    '{"mental_models": [{"model_name": "validation", '
    '"description": "seeking support", "probability": 1.0}]}'
)


def test_clean_completion_and_nonempty_response():
    parsed = parse_completion(f"{BLOCK}\nRESPONSE:\nA useful reply")
    assert parsed.status == CLEAN
    assert parsed.has_response is True
    assert parsed.response == "A useful reply"


def test_fenced_or_prefaced_json_is_marked_repaired():
    parsed = parse_completion(
        f"Here are the assumptions:\n```json\n{BLOCK}\n```\nRESPONSE:\nreply"
    )
    assert parsed.status == REPAIRED
    assert parsed.n_assumptions == 1


def test_assumptions_after_reply_are_rejected_from_usable_set():
    parsed = parse_completion(f"RESPONSE:\nreply first\n{BLOCK}")
    assert parsed.status == INVALID_ORDER
    assert parsed.has_response is True


def test_empty_response_heading_is_not_a_reply():
    parsed = parse_completion(f"{BLOCK}\nRESPONSE:\n   ")
    assert parsed.has_response is False
    assert "empty RESPONSE" in parsed.notes


def test_numbered_markdown_fallback_and_expected_count_diagnostic():
    raw = """1. **Model name:** seeking support
**Description:** The user wants reassurance.
**Probability:** 70%

2. **Model name:** weighing options
**Description:** The user wants an outside view.
**Probability:** 30%

ANSWER:
A useful reply"""
    parsed = parse_completion(raw, expected_n=3)
    assert parsed.status == SALVAGED
    assert [m.assumption for m in parsed.mental_models] == [
        "seeking support", "weighing options",
    ]
    assert parsed.prob_sum == 1.0
    assert parsed.response == "A useful reply"
    assert "expected 3 assumptions, found 2" in parsed.notes


def test_probe_label_names_the_papers_prompt_type():
    """No invented framing in the label -- see tests/test_prompts.py for the
    verbatim diff against the vendored reference implementation."""
    assert ProbeSpec().label() == "openended3"
    assert ProbeSpec().family == "open-ended"


FOUR_DIMS = {
    "validation_seeking": {"score": 0.8, "explanation": "Wants reassurance."},
    "user_rightness": {"score": 0.6, "explanation": "May be right."},
    "user_information_advantage": {"score": 0.7, "explanation": "Was there."},
    "objectivity_seeking": {"score": 0.4, "explanation": "Asks for judgment."},
}


def _structured_raw(payload=FOUR_DIMS):
    block = {"mental_model": {"beliefs": payload}}
    return f"{json.dumps(block)}\nRESPONSE:\nA useful reply"


def test_clean_structured_completion_keeps_fixed_dimension_order():
    parsed = parse_structured(_structured_raw(), "4dims")
    assert parsed.status == CLEAN
    assert parsed.n_dimensions == 4
    assert parsed.n_scored == 4
    assert [belief.dimension for belief in parsed.beliefs] == list(FOUR_DIMS)
    assert [belief.score for belief in parsed.beliefs] == [0.8, 0.6, 0.7, 0.4]
    assert parsed.response == "A useful reply"

    failed = parse_structured("", "4dims")
    assert failed.status == "failed"
    assert failed.n_dimensions == 4
    assert failed.n_scored == 0
    assert [belief.dimension for belief in failed.beliefs] == list(FOUR_DIMS)


def test_structured_parser_marks_repairs_missing_scores_and_invalid_order():
    fenced = parse_structured(
        f"Here it is:\n```json\n{json.dumps({'mental_model': {'beliefs': FOUR_DIMS}})}\n```"
        "\nRESPONSE:\nreply",
        "4dims",
    )
    assert fenced.status == REPAIRED

    partial = dict(FOUR_DIMS)
    partial.pop("objectivity_seeking")
    partial["user_rightness"] = {"score": 1.2, "explanation": "invalid"}
    salvaged = parse_structured(_structured_raw(partial), "4dims")
    assert salvaged.status == SALVAGED
    assert salvaged.n_scored == 2
    assert "invalid score(s): user_rightness" in salvaged.notes

    after = parse_structured(
        "RESPONSE:\nreply first\n" + json.dumps({"mental_model": {"beliefs": FOUR_DIMS}}),
        "4dims",
    )
    assert after.status == INVALID_ORDER


def test_structured_records_have_lean_and_full_schemas():
    parsed = parse_structured(_structured_raw(), "4dims")
    source = {
        "run_id": "run-1", "probe": "4dims", "persona_type": "age",
        "persona_id": "p1", "prompt_type": "original_post",
        "prompt_id": "q1", "rep": 0,
    }
    lean = to_structured_records(source, parsed)
    full = to_structured_records(source, parsed, full=True)
    assert len(lean) == len(full) == 4
    assert set(lean[0]) == {
        "run_id", "probe", "persona_type", "persona_id", "prompt_type",
        "prompt_id", "rep", "dimension", "score", "explanation",
        "parse_status", "has_response",
    }
    assert "n_scored" in full[0]


def test_parse_command_routes_structured_output_by_probe_label(tmp_path):
    raw = tmp_path / "run.jsonl"
    row = {
        "cell_key": "cell-1", "run_id": "run-1", "probe": "4dims",
        "persona_type": "age", "persona_id": "p1",
        "prompt_type": "original_post", "prompt_id": "q1", "rep": 0,
        "n_dimensions_asked": 4, "raw": _structured_raw(), "error": "",
    }
    raw.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert parse_main([str(raw), "--format", "csv"]) == 0
    output = tmp_path / "run_structured.csv"
    parsed = pd.read_csv(output)
    assert len(parsed) == 4
    assert set(parsed.dimension) == set(FOUR_DIMS)


def test_redundant_model_columns_are_not_in_raw_row_schema():
    columns = {field.name for field in fields(AssumptionRecord)}
    assert not columns.intersection({
        "model_id", "model_family", "model_generation", "model_release_date",
        "backend", "quantized_file",
    })
    assert AssumptionRecord.__dataclass_fields__["schema_version"].default == 3


def test_parsed_rows_drop_redundant_legacy_model_columns_too():
    parsed = parse_completion(f"{BLOCK}\nRESPONSE:\nreply")
    legacy = {
        "run_id": "run-1",
        "model_id": "old-model",
        "model_family": "old-family",
        "model_generation": "old-generation",
        "model_release_date": "2000-01-01",
        "backend": "old-backend",
        "quantized_file": "old.gguf",
    }
    for full in (False, True):
        row = to_records(legacy, parsed, full=full)[0]
        assert not set(legacy).difference({"run_id"}).intersection(row)


def test_latest_success_wins_over_failed_retries():
    attempts = [
        {"cell_key": "a", "error": "timeout", "raw": ""},
        {"cell_key": "a", "error": "", "raw": "success"},
        {"cell_key": "a", "error": "timeout again", "raw": ""},
        {"cell_key": "b", "error": "first", "raw": ""},
        {"cell_key": "b", "error": "last", "raw": ""},
    ]
    rows, diagnostics = canonical_rows(attempts)
    indexed = {row["cell_key"]: row for row in rows}
    assert indexed["a"]["raw"] == "success"
    assert indexed["b"]["error"] == "last"
    assert diagnostics == {
        "attempts": 5,
        "cells": 2,
        "retried_cells": 2,
        "extra_attempts": 3,
    }
