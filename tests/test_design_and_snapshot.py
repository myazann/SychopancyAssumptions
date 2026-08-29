import json

import pytest

from syco.design import coordinate_digest, digest, selection_for, target_coordinates
from syco.snapshot import sha256_file, verify


def _write_design(path, persona_path, prompt_path):
    factors = {
        "persona_types": ["facet"],
        "prompt_types": ["original_post", "flipped_story"],
        "n_reps": 1,
        "include_control": True,
        "control": {"persona_type": "none", "persona_id": "none"},
    }
    selection = {"persona_ids": ["p1"], "prompt_ids": ["q1"]}
    coordinates = target_coordinates(
        persona_ids=selection["persona_ids"],
        persona_types=factors["persona_types"],
        prompt_ids=selection["prompt_ids"],
        prompt_types=factors["prompt_types"],
        n_reps=1,
        include_control=True,
        control_type="none",
        control_id="none",
    )
    identity = {
        "name": "fixture",
        "data": {
            "personas_sha256": sha256_file(persona_path),
            "prompts_sha256": sha256_file(prompt_path),
        },
        "selection": selection,
        "factors": factors,
        "expected_coordinates": len(coordinates),
        "coordinates_sha256": coordinate_digest(coordinates),
        "instruments": [{"probe": "4dims", "expected_cells": len(coordinates)}],
    }
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "syco-design-lock",
                "design_id": digest(identity),
                "identity": identity,
            }
        ),
        encoding="utf-8",
    )


def test_frozen_design_verifies_data_and_returns_exact_selection(tmp_path):
    personas = tmp_path / "personas.gz"
    prompts = tmp_path / "prompts.gz"
    personas.write_bytes(b"personas")
    prompts.write_bytes(b"prompts")
    design = tmp_path / "design.json"
    _write_design(design, personas, prompts)

    selected = selection_for(
        design, probe="4dims", persona_path=personas, prompt_path=prompts
    )
    assert selected["selection"] == {
        "persona_ids": ["p1"],
        "prompt_ids": ["q1"],
    }

    prompts.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="prompt data"):
        selection_for(
            design, probe="4dims", persona_path=personas, prompt_path=prompts
        )


def test_snapshot_verifier_detects_tampering(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    artifact = snapshot / "artifact.txt"
    artifact.write_text("original", encoding="utf-8")
    (snapshot / "checksums.sha256").write_text(
        f"{sha256_file(artifact)}  artifact.txt\n", encoding="utf-8"
    )

    assert verify(snapshot) == 0
    artifact.write_text("changed", encoding="utf-8")
    assert verify(snapshot) == 1


def _run_output(path, *, probe, persona_ids, prompt_ids, run_id):
    from syco.design import target_coordinates as coordinates_for

    coordinates = coordinates_for(
        persona_ids=persona_ids,
        persona_types=["a", "b"],
        prompt_ids=prompt_ids,
        prompt_types=["original_post", "flipped_story"],
        n_reps=1,
        include_control=True,
        control_type="none",
        control_id="none",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "cell_key": "|".join((run_id, *map(str, coordinate))),
            "run_id": run_id,
            "probe": probe,
            "persona_type": coordinate[0],
            "persona_id": coordinate[1],
            "prompt_type": coordinate[2],
            "prompt_id": coordinate[3],
            "rep": coordinate[4],
            "error": "",
            "prompt_digest": "digest",
        }
        for coordinate in sorted(coordinates)
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    manifest = {
        "schema_version": 2,
        "run_id": run_id,
        "identity": {
            "instrument": {"probe": probe, "system": "", "thinking": False},
            "design": {
                "persona_types": ["a", "b"],
                "prompt_types": ["original_post", "flipped_story"],
                "n_reps": 1,
                "include_control": True,
                "seed": 1000,
            },
            "data": {"personas_sha256": "p", "prompts_sha256": "q"},
        },
    }
    path.with_name(path.name + ".manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return len(coordinates)


def test_freeze_locks_the_design_a_finished_run_already_administered(tmp_path):
    """Starts the wave chain from a run you already have, rather than from an
    extension that does not exist yet."""
    from syco.design import freeze_design_lock, verify_design_lock

    four = tmp_path / "4dims.jsonl"
    support = tmp_path / "supporttypes.jsonl"
    cells = _run_output(
        four, probe="4dims", persona_ids=["p1", "p2"], prompt_ids=["q1"], run_id="r1"
    )
    _run_output(
        support,
        probe="supporttypes",
        persona_ids=["p1", "p2"],
        prompt_ids=["q1"],
        run_id="r2",
    )

    lock = freeze_design_lock("base", [four, support])
    output = tmp_path / "base.json"
    output.write_text(json.dumps(lock), encoding="utf-8")

    summary = verify_design_lock(output)
    assert summary["personas"] == 2
    assert summary["prompts"] == 1
    assert summary["coordinates_per_instrument"] == cells
    assert summary["instruments"] == ["4dims", "supporttypes"]
    assert lock["identity"]["selection"]["persona_ids"] == ["p1", "p2"]


def test_freeze_refuses_a_run_that_is_not_a_complete_grid(tmp_path):
    from syco.design import freeze_design_lock

    output = tmp_path / "partial.jsonl"
    _run_output(
        output, probe="4dims", persona_ids=["p1"], prompt_ids=["q1"], run_id="r1"
    )
    lines = output.read_text(encoding="utf-8").splitlines()[:-1]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="not a complete paired grid"):
        freeze_design_lock("base", [output])


def test_every_checked_in_design_lock_verifies():
    """A lock is the study. A broken one must not reach a job submission."""
    from syco import paths
    from syco.design import verify_design_lock

    locks = sorted((paths.CONFIG_DIR / "designs").glob("*.json"))
    assert locks, "no design locks are checked in"
    for lock in locks:
        summary = verify_design_lock(lock)
        assert summary["personas"] > 0
        assert summary["prompts"] > 0
        assert summary["instruments"]


def test_every_wave_profile_names_a_lock_that_covers_its_probe():
    """Catches a profile pointed at the wrong instrument's design."""
    from syco import paths
    from syco.design import read_json
    from syco.experiments import load_profile

    for path in sorted((paths.CONFIG_DIR / "experiments").glob("*.yaml")):
        profile = load_profile(str(path))
        if profile.design_path is None:
            continue
        identity = read_json(profile.design_path)["identity"]
        probes = {item.get("probe") for item in identity["instruments"]}
        assert profile.probe_spec.kind in probes, (
            f"{path.name} runs {profile.probe_spec.kind!r} but its lock covers "
            f"{sorted(probes)}"
        )
