import json

import pytest

from syco.data import Persona, Prompt
from syco.extensions import (
    cell_coordinate,
    combine_extension,
    plan_extension,
    row_coordinate,
)
from syco.grid import build_cells
from syco.manifest import load_manifest, write_manifest
from syco.store import read_rows


def _persona(facet, person):
    return Persona(person, facet, (), False, 0)


def _fixture_design():
    personas = [
        _persona(facet, person) for person in ("p1", "p2", "p3") for facet in ("a", "b")
    ]
    prompts = [
        Prompt(prompt_id, framing, framing)
        for prompt_id in ("q1", "q2", "q3")
        for framing in ("original_post", "flipped_story")
    ]
    return personas, prompts


def _identity(design):
    return {
        "model": {"alias": "model", "ref": "model-ref"},
        "instrument": {"probe": "4dims", "system": "", "thinking": False},
        "design": design,
        "data": {"personas_sha256": "p", "prompts_sha256": "q"},
    }


def _row(cell, run_id):
    persona_type, persona_id, prompt_type, prompt_id, rep = cell_coordinate(cell)
    return {
        "cell_key": "|".join((run_id, *map(str, cell_coordinate(cell)))),
        "run_id": run_id,
        "probe": "4dims",
        "persona_type": persona_type,
        "persona_id": persona_id,
        "prompt_type": prompt_type,
        "prompt_id": prompt_id,
        "rep": rep,
        "error": "",
        "raw": "completion",
        "prompt_digest": "digest",
    }


def _write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_extension_is_disjoint_and_fills_the_complete_union_grid(tmp_path):
    personas, prompts = _fixture_design()
    base_cells = build_cells(
        personas,
        prompts,
        persona_types=["a", "b"],
        prompt_types=["original_post", "flipped_story"],
        persona_ids=["p1"],
        prompt_ids=["q1"],
    )
    base = tmp_path / "base.jsonl"
    _write_rows(base, [_row(cell, "base") for cell in base_cells])
    write_manifest(
        base,
        {
            "schema_version": 1,
            "run_id": "base",
            "identity": _identity(
                {
                    "persona_types": ["a", "b"],
                    "prompt_types": ["original_post", "flipped_story"],
                    "n_personas": 1,
                    "n_prompts": 1,
                    "n_reps": 1,
                    "include_control": True,
                    "seed": 1,
                }
            ),
        },
    )

    plan = plan_extension(
        base,
        personas,
        prompts,
        persona_types=["a", "b"],
        prompt_types=["original_post", "flipped_story"],
        additional_personas=1,
        additional_prompts=1,
        include_no_persona=True,
        n_reps=1,
        seed=7,
    )
    assert not (set(plan.base_persona_ids) & set(plan.added_persona_ids))
    assert not (set(plan.base_prompt_ids) & set(plan.added_prompt_ids))
    assert len(plan.base_coordinates) == 6
    assert len(plan.cells) == 14
    assert not (set(plan.base_coordinates) & {cell_coordinate(c) for c in plan.cells})

    union = set(plan.base_coordinates) | {cell_coordinate(cell) for cell in plan.cells}
    assert len(union) == (2 * 2 + 1) * 2 * 2


def test_extension_uses_exact_frozen_target_ids(tmp_path):
    personas, prompts = _fixture_design()
    base_cells = build_cells(
        personas,
        prompts,
        persona_types=["a", "b"],
        prompt_types=["original_post", "flipped_story"],
        persona_ids=["p1"],
        prompt_ids=["q1"],
    )
    base = tmp_path / "base.jsonl"
    _write_rows(base, [_row(cell, "base") for cell in base_cells])
    write_manifest(
        base,
        {
            "schema_version": 1,
            "run_id": "base",
            "identity": _identity(
                {
                    "persona_types": ["a", "b"],
                    "prompt_types": ["original_post", "flipped_story"],
                    "n_personas": 1,
                    "n_prompts": 1,
                    "n_reps": 1,
                    "include_control": True,
                    "seed": 1,
                }
            ),
        },
    )

    plan = plan_extension(
        base,
        personas,
        prompts,
        persona_types=["a", "b"],
        prompt_types=["original_post", "flipped_story"],
        additional_personas=1,
        additional_prompts=1,
        include_no_persona=True,
        n_reps=1,
        seed=999,
        target_persona_ids=["p1", "p3"],
        target_prompt_ids=["q1", "q3"],
    )

    assert plan.added_persona_ids == ("p3",)
    assert plan.added_prompt_ids == ("q3",)
    assert plan.all_persona_ids == ("p1", "p3")
    assert plan.all_prompt_ids == ("q1", "q3")


def test_collection_combines_shards_under_one_analysis_run_id(tmp_path):
    personas, prompts = _fixture_design()
    base_cells = build_cells(
        personas,
        prompts,
        persona_types=["a", "b"],
        prompt_types=["original_post", "flipped_story"],
        persona_ids=["p1"],
        prompt_ids=["q1"],
    )
    base = tmp_path / "base.jsonl"
    base_design = {
        "persona_types": ["a", "b"],
        "prompt_types": ["original_post", "flipped_story"],
        "n_personas": 1,
        "n_prompts": 1,
        "n_reps": 1,
        "include_control": True,
        "seed": 1,
    }
    _write_rows(base, [_row(cell, "base") for cell in base_cells])
    base_manifest = {
        "schema_version": 1,
        "run_id": "base",
        "identity": _identity(base_design),
    }
    write_manifest(base, base_manifest)
    plan = plan_extension(
        base,
        personas,
        prompts,
        persona_types=["a", "b"],
        prompt_types=["original_post", "flipped_story"],
        additional_personas=1,
        additional_prompts=1,
        include_no_persona=True,
        n_reps=1,
        seed=7,
    )

    extension = tmp_path / "extension.jsonl"
    _write_rows(extension, [_row(cell, "extension") for cell in plan.cells])
    extension_design = dict(base_design)
    extension_design.update(n_personas=1, n_prompts=1, seed=7, extension=plan.identity)
    write_manifest(
        extension,
        {
            "schema_version": 1,
            "run_id": "extension",
            "identity": _identity(extension_design),
        },
    )

    target = combine_extension(base, extension, tmp_path / "collection.jsonl")
    rows = read_rows(target)
    manifest = load_manifest(target)
    assert len(rows) == 20
    assert {row["run_id"] for row in rows} == {manifest["run_id"]}
    assert {row["source_run_id"] for row in rows} == {"base", "extension"}
    assert len({row_coordinate(row) for row in rows}) == 20
    assert manifest["identity"]["design"]["n_personas"] == 2
    assert manifest["identity"]["design"]["n_prompts"] == 2


def _write_shard(path, cells, run_id, design):
    _write_rows(path, [_row(cell, run_id) for cell in cells])
    write_manifest(
        path,
        {"schema_version": 2, "run_id": run_id, "identity": _identity(design)},
    )


def _base_design(**overrides):
    design = {
        "persona_types": ["a", "b"],
        "prompt_types": ["original_post", "flipped_story"],
        "n_personas": 1,
        "n_prompts": 1,
        "n_reps": 1,
        "include_control": True,
        "seed": 1,
    }
    design.update(overrides)
    return design


def _cells(personas, prompts, persona_ids, prompt_ids):
    return build_cells(
        personas,
        prompts,
        persona_types=["a", "b"],
        prompt_types=["original_post", "flipped_story"],
        persona_ids=persona_ids,
        prompt_ids=prompt_ids,
    )


def test_a_third_wave_extends_every_earlier_shard_at_once(tmp_path):
    """The additive case: each wave only runs what no earlier wave collected."""
    personas, prompts = _fixture_design()
    base = tmp_path / "base.jsonl"
    _write_shard(base, _cells(personas, prompts, ["p1"], ["q1"]), "base", _base_design())

    first = plan_extension(
        base,
        personas,
        prompts,
        persona_types=None,
        prompt_types=None,
        additional_personas=None,
        additional_prompts=None,
        include_no_persona=True,
        n_reps=1,
        seed=1,
        target_persona_ids=["p1", "p2"],
        target_prompt_ids=["q1", "q2"],
    )
    wave_one = tmp_path / "wave1.jsonl"
    _write_shard(
        wave_one,
        first.cells,
        "wave1",
        _base_design(cells=len(first.cells), extension=first.identity),
    )

    second = plan_extension(
        [base, wave_one],
        personas,
        prompts,
        persona_types=None,
        prompt_types=None,
        additional_personas=1,
        additional_prompts=1,
        include_no_persona=True,
        n_reps=1,
        seed=1,
        target_persona_ids=["p1", "p2", "p3"],
        target_prompt_ids=["q1", "q2", "q3"],
    )

    covered = set(first.base_coordinates) | {cell_coordinate(c) for c in first.cells}
    assert set(second.base_coordinates) == covered
    assert second.added_persona_ids == ("p3",)
    # 3 people x 2 facets, plus one control, x 3 dilemmas x 2 framings.
    assert len(second.target_coordinates) == (3 * 2 + 1) * 3 * 2
    assert len(second.cells) == len(second.target_coordinates) - len(covered)
    assert not (covered & {cell_coordinate(c) for c in second.cells})


def test_an_unfinished_shard_cannot_be_built_on(tmp_path):
    """Otherwise two jobs would write the same coordinates into two files."""
    personas, prompts = _fixture_design()
    cells = _cells(personas, prompts, ["p1"], ["q1"])
    partial = tmp_path / "partial.jsonl"
    _write_shard(partial, cells[:-2], "partial", _base_design(cells=len(cells)))

    with pytest.raises(RuntimeError, match="still incomplete"):
        plan_extension(
            partial,
            personas,
            prompts,
            persona_types=None,
            prompt_types=None,
            additional_personas=None,
            additional_prompts=None,
            include_no_persona=True,
            n_reps=1,
            seed=1,
            target_persona_ids=["p1", "p2"],
            target_prompt_ids=["q1", "q2"],
        )


def test_unset_persona_types_match_a_base_that_recorded_none(tmp_path):
    """`persona_types: null` means "every facet" on both sides of the comparison.

    Deriving the base's facets from its rows while leaving the caller's `None`
    alone made this comparison impossible to satisfy, which took `syco status`
    down for every extension profile.
    """
    personas, prompts = _fixture_design()
    base = tmp_path / "base.jsonl"
    _write_shard(
        base,
        _cells(personas, prompts, ["p1"], ["q1"]),
        "base",
        _base_design(persona_types=None, prompt_types=None),
    )

    plan = plan_extension(
        base,
        personas,
        prompts,
        persona_types=None,
        prompt_types=None,
        additional_personas=1,
        additional_prompts=1,
        include_no_persona=True,
        n_reps=1,
        seed=3,
    )

    assert len(plan.cells) == 14


def test_shards_that_disagree_with_the_target_are_refused(tmp_path):
    personas, prompts = _fixture_design()
    base = tmp_path / "base.jsonl"
    _write_shard(base, _cells(personas, prompts, ["p1"], ["q1"]), "base", _base_design())

    with pytest.raises(RuntimeError, match="does not contain everything already"):
        plan_extension(
            base,
            personas,
            prompts,
            persona_types=None,
            prompt_types=None,
            additional_personas=None,
            additional_prompts=None,
            include_no_persona=True,
            n_reps=1,
            seed=1,
            target_persona_ids=["p2", "p3"],
            target_prompt_ids=["q2", "q3"],
        )


def _lock(path, persona_ids, prompt_ids):
    from syco.design import coordinate_digest, digest, target_coordinates

    factors = {
        "persona_types": ["a", "b"],
        "prompt_types": ["original_post", "flipped_story"],
        "n_reps": 1,
        "include_control": True,
        "control": {"persona_type": "none", "persona_id": "none"},
    }
    coordinates = target_coordinates(
        persona_ids=persona_ids,
        persona_types=factors["persona_types"],
        prompt_ids=prompt_ids,
        prompt_types=factors["prompt_types"],
        n_reps=1,
        include_control=True,
        control_type="none",
        control_id="none",
    )
    identity = {
        "name": "fixture",
        "data": {"personas_sha256": "p", "prompts_sha256": "q"},
        "selection": {"persona_ids": persona_ids, "prompt_ids": prompt_ids},
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
    return path


def test_a_collection_is_checked_against_the_design_it_should_fill(tmp_path):
    """The self-consistency check cannot see a facet missing from every wave."""
    from syco.extensions import collect

    personas, prompts = _fixture_design()
    base = tmp_path / "base.jsonl"
    _write_shard(base, _cells(personas, prompts, ["p1"], ["q1"]), "base", _base_design())
    plan = plan_extension(
        base,
        personas,
        prompts,
        persona_types=None,
        prompt_types=None,
        additional_personas=None,
        additional_prompts=None,
        include_no_persona=True,
        n_reps=1,
        seed=1,
        target_persona_ids=["p1", "p2"],
        target_prompt_ids=["q1", "q2"],
    )
    wave = tmp_path / "wave.jsonl"
    _write_shard(
        wave, plan.cells, "wave", _base_design(cells=len(plan.cells),
                                               extension=plan.identity)
    )
    lock = _lock(tmp_path / "lock.json", ["p1", "p2"], ["q1", "q2"])

    target = collect([base, wave], tmp_path / "all.jsonl", design_lock=lock,
                     probe="4dims")
    rows = read_rows(target)

    # 2 people x 2 facets, plus one control, x 2 dilemmas x 2 framings.
    assert len(rows) == 20
    assert len({row["run_id"] for row in rows}) == 1
    assert len({row["source_run_id"] for row in rows}) == 2


def test_a_collection_missing_a_wave_is_refused(tmp_path):
    from syco.extensions import collect

    personas, prompts = _fixture_design()
    base = tmp_path / "base.jsonl"
    _write_shard(base, _cells(personas, prompts, ["p1"], ["q1"]), "base", _base_design())
    other = tmp_path / "other.jsonl"
    _write_shard(
        other,
        _cells(personas, prompts, ["p2"], ["q2"]),
        "other",
        _base_design(cells=6),
    )
    lock = _lock(tmp_path / "lock.json", ["p1", "p2"], ["q1", "q2"])

    with pytest.raises(RuntimeError, match="the frozen design expects"):
        collect([base, other], tmp_path / "all.jsonl", design_lock=lock, probe="4dims")


def test_a_wave_crosses_the_original_people_with_the_new_dilemmas(tmp_path):
    """The scientifically load-bearing property of a wave.

    A wave is not "another block of cells". Its largest part is the people
    already collected, now answering the dilemmas just added -- without which
    the design is two disjoint blocks and the person-level contrast cannot be
    read across the whole grid.
    """
    personas, prompts = _fixture_design()
    base = tmp_path / "base.jsonl"
    _write_shard(
        base, _cells(personas, prompts, ["p1", "p2"], ["q1"]), "base", _base_design()
    )

    plan = plan_extension(
        base,
        personas,
        prompts,
        persona_types=None,
        prompt_types=None,
        additional_personas=1,
        additional_prompts=2,
        include_no_persona=True,
        n_reps=1,
        seed=1,
        target_persona_ids=["p1", "p2", "p3"],
        target_prompt_ids=["q1", "q2", "q3"],
    )

    old_people, new_people = {"p1", "p2"}, {"p3"}
    new_dilemmas = {"q2", "q3"}
    emitted = {cell_coordinate(cell) for cell in plan.cells}

    old_x_new = {
        (persona_id, prompt_id)
        for _, persona_id, _, prompt_id, _ in emitted
        if persona_id in old_people and prompt_id in new_dilemmas
    }
    assert old_x_new == {(p, q) for p in old_people for q in new_dilemmas}

    # Both of the other blocks are present too, and nothing is re-run.
    assert any(p in new_people and q == "q1" for _, p, _, q, _ in emitted)
    assert any(p in new_people and q in new_dilemmas for _, p, _, q, _ in emitted)
    assert not (emitted & set(plan.base_coordinates))

    # Every person ends up against every dilemma, in every facet and framing.
    union = set(plan.base_coordinates) | emitted
    assert union == set(plan.target_coordinates)
    seen = {}
    for persona_type, persona_id, prompt_type, prompt_id, _ in union:
        if persona_id == "none":
            continue
        seen.setdefault(persona_id, set()).add((persona_type, prompt_type, prompt_id))
    assert {len(v) for v in seen.values()} == {2 * 2 * 3}
