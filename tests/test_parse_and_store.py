from dataclasses import fields

from syco.parse import (
    CLEAN,
    INVALID_ORDER,
    REPAIRED,
    SALVAGED,
    parse_completion,
    to_records,
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


def test_redundant_model_columns_are_not_in_raw_row_schema():
    columns = {field.name for field in fields(AssumptionRecord)}
    assert not columns.intersection({
        "model_id", "model_family", "model_generation", "model_release_date",
        "backend", "quantized_file",
    })
    assert AssumptionRecord.__dataclass_fields__["schema_version"].default == 2


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
