import pytest

from syco.manifest import (
    acquisition_digest,
    ensure_manifest,
    identity_conflicts,
    load_manifest,
    reconcile_manifest,
    repo_digest,
)


def _manifest(run_id, *, probe="4dims", alias="model", coordinates=None, digest="a"):
    return {
        "schema_version": 2,
        "run_id": run_id,
        "identity": {
            "model": {"alias": alias, "ref": "ref", "quantization": "q"},
            "instrument": {"probe": probe, "system": "", "thinking": False},
            "design": {
                "persona_types": None,
                "prompt_types": ["original_post"],
                "n_reps": 1,
                "include_control": True,
                "coordinates_sha256": coordinates,
            },
            "data": {"personas_sha256": "p", "prompts_sha256": "q"},
            "acquisition_digest": digest,
        },
    }


def test_manifest_is_written_once(tmp_path):
    output = tmp_path / "run.jsonl"
    first = _manifest("first")

    manifest, adopted = reconcile_manifest(output, first, has_output=False)
    assert manifest == first
    assert adopted is False
    assert load_manifest(output) == first


def test_an_incompatible_configuration_is_rejected(tmp_path):
    output = tmp_path / "run.jsonl"
    ensure_manifest(output, _manifest("first"), has_output=False)

    with pytest.raises(RuntimeError, match="instrument.probe"):
        ensure_manifest(output, _manifest("second", probe="supporttypes"), has_output=True)

    with pytest.raises(RuntimeError, match="model.alias"):
        ensure_manifest(output, _manifest("third", alias="other"), has_output=True)


def test_a_changed_design_is_rejected(tmp_path):
    output = tmp_path / "run.jsonl"
    ensure_manifest(output, _manifest("first", coordinates="aaa"), has_output=False)

    with pytest.raises(RuntimeError, match="design.coordinates_sha256"):
        ensure_manifest(output, _manifest("second", coordinates="bbb"), has_output=True)


def test_a_run_continues_under_its_recorded_identity_after_code_moves_on(tmp_path):
    """The point of the whole mechanism: a long study survives its own edits.

    `cell_key` embeds `run_id`, so recomputing a new one would orphan every row
    already collected. When nothing that changes an observation has moved, the
    recorded identity stays authoritative and the drift is appended instead.
    """
    output = tmp_path / "run.jsonl"
    ensure_manifest(output, _manifest("first", digest="before"), has_output=False)

    later = _manifest("second", digest="after")
    manifest, adopted = reconcile_manifest(output, later, has_output=True)

    assert adopted is True
    assert manifest["run_id"] == "first"
    assert manifest["revisions"][-1]["acquisition_digest"] == "after"
    assert load_manifest(output)["run_id"] == "first"


def test_repeated_reconciliation_records_one_revision(tmp_path):
    output = tmp_path / "run.jsonl"
    ensure_manifest(output, _manifest("first", digest="before"), has_output=False)
    later = _manifest("second", digest="after")

    reconcile_manifest(output, later, has_output=True)
    reconcile_manifest(output, later, has_output=True)

    assert len(load_manifest(output)["revisions"]) == 1


def test_a_missing_field_on_either_side_is_not_a_conflict():
    """Manifests written before schema 2 have no coordinate digest.

    Treating their absence as a difference would strand every output collected
    so far, which is exactly what this change exists to prevent.
    """
    legacy = {"identity": {"design": {"persona_types": None, "n_reps": 1}}}
    current = _manifest("new", coordinates="aaa")

    assert identity_conflicts(legacy, current) == []


def test_plan_only_leaves_no_manifest_behind(tmp_path):
    output = tmp_path / "run.jsonl"

    manifest, adopted = reconcile_manifest(
        output, _manifest("first"), has_output=False, write=False
    )

    assert manifest["run_id"] == "first"
    assert adopted is False
    assert load_manifest(output) is None


def test_legacy_nonempty_output_cannot_be_resumed_without_manifest(tmp_path):
    output = tmp_path / "legacy.jsonl"
    output.write_text("{}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="has no run manifest"):
        ensure_manifest(output, _manifest("new"), has_output=True)


def test_analysis_code_is_outside_the_acquisition_digest():
    """The digest that drives run_id must not move when analysis code does."""
    before = acquisition_digest()
    repo_before = repo_digest()

    from syco import paths

    scratch = paths.ROOT / "syco" / "_digest_probe.py"
    scratch.write_text("# temporary\n", encoding="utf-8")
    try:
        assert acquisition_digest() == before
        assert repo_digest() != repo_before
    finally:
        scratch.unlink()
