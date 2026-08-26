import pytest

from syco.manifest import ensure_manifest, load_manifest


def test_manifest_is_written_once_and_mismatch_is_rejected(tmp_path):
    output = tmp_path / "run.jsonl"
    first = {"schema_version": 1, "run_id": "first", "identity": {"x": 1}}
    second = {"schema_version": 1, "run_id": "second", "identity": {"x": 2}}

    ensure_manifest(output, first, has_output=False)
    assert load_manifest(output) == first
    ensure_manifest(output, first, has_output=False)
    with pytest.raises(RuntimeError, match="belongs to run first"):
        ensure_manifest(output, second, has_output=False)


def test_legacy_nonempty_output_cannot_be_resumed_without_manifest(tmp_path):
    output = tmp_path / "legacy.jsonl"
    output.write_text("{}\n", encoding="utf-8")
    expected = {"schema_version": 1, "run_id": "new", "identity": {}}

    with pytest.raises(RuntimeError, match="has no run manifest"):
        ensure_manifest(output, expected, has_output=True)
