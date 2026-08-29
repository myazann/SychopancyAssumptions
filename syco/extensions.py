"""Collect a study in waves without confounding the design.

A wave is an immutable acquisition shard. It reads every coordinate already
collected in one or more prior shards, works out the full target grid, and
emits *only* the coordinates that target still lacks. For a 20 x 20 base plus
20 new people and 20 new dilemmas that is the missing 24,040 cells -- the
old-person/new-dilemma, new-person/old-dilemma, and new-person/new-dilemma
blocks -- not a confounded second 20 x 20 block.

The target comes from a frozen design lock whenever one is given, which is what
makes waves composable: wave three names the same lock and lists waves one and
two as its sources, and the planner works out the remainder. Without a lock the
older single-base form still applies, where `additional_personas` and
`additional_prompts` draw fresh IDs from the unused eligible pool.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from syco.data import NO_PERSONA
from syco.grid import Cell, build_cells, eligible_design_ids, stable_sample
from syco.manifest import load_manifest, write_manifest
from syco.store import canonical_rows, read_rows

Coordinate = tuple[str, str, str, str, int]


def row_coordinate(row: dict) -> Coordinate:
    return (
        str(row.get("persona_type")),
        str(row.get("persona_id")),
        str(row.get("prompt_type")),
        str(row.get("prompt_id")),
        int(row.get("rep", 0)),
    )


def cell_coordinate(cell: Cell) -> Coordinate:
    return (*cell.key_parts[:-1], int(cell.rep))


def _ordered_unique(values) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values))


def _coordinates_digest(coordinates) -> str:
    canonical = json.dumps(sorted(coordinates), separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _as_paths(value) -> list[Path]:
    """One output or several, always in the order the caller wrote them."""
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        value = [value]
    elif not isinstance(value, Sequence):
        raise RuntimeError(  # noqa: TRY004
            f"expected a path or a sequence of paths: {value!r}"
        )
    seen: dict[Path, None] = {}
    for item in value:
        seen.setdefault(Path(item).resolve(), None)
    return list(seen)


@dataclass(frozen=True)
class Source:
    """One prior shard, validated and reduced to what a plan needs from it."""

    path: Path
    manifest: dict
    rows: tuple[dict, ...]
    coordinates: frozenset[Coordinate]

    @property
    def record(self) -> dict:
        return {
            "path": str(self.path),
            "run_id": self.manifest.get("run_id"),
            "cells": len(self.coordinates),
            "coordinates_sha256": _coordinates_digest(set(self.coordinates)),
        }


@dataclass(frozen=True)
class ExtensionPlan:
    sources: tuple[Source, ...]
    base_rows: tuple[dict, ...]
    base_coordinates: frozenset[Coordinate]
    base_persona_ids: tuple[str, ...]
    base_prompt_ids: tuple[str, ...]
    added_persona_ids: tuple[str, ...]
    added_prompt_ids: tuple[str, ...]
    all_persona_ids: tuple[str, ...]
    all_prompt_ids: tuple[str, ...]
    target_coordinates: frozenset[Coordinate]
    cells: tuple[Cell, ...]

    @property
    def base_path(self) -> Path:
        return self.sources[0].path

    @property
    def base_manifest(self) -> dict:
        return self.sources[0].manifest

    @property
    def identity(self) -> dict:
        value = {
            "mode": "full-cross-extension",
            "sources": [source.record for source in self.sources],
            "covered_cells": len(self.base_coordinates),
            "target_cells": len(self.target_coordinates),
            "base_persona_ids": list(self.base_persona_ids),
            "base_prompt_ids": list(self.base_prompt_ids),
            "added_persona_ids": list(self.added_persona_ids),
            "added_prompt_ids": list(self.added_prompt_ids),
            "total_persona_ids": len(self.all_persona_ids),
            "total_prompt_ids": len(self.all_prompt_ids),
            "extension_cells": len(self.cells),
        }
        if len(self.sources) == 1:
            # Keep the single-source shape that design locks and collections
            # written before multi-source waves already read.
            source = self.sources[0]
            value["base_output"] = str(source.path)
            value["base_run_id"] = source.manifest.get("run_id")
            value["base_coordinates_sha256"] = source.record["coordinates_sha256"]
            value["base_cells"] = len(source.coordinates)
        return value


def _manifest_design(manifest: dict) -> dict:
    return dict((manifest.get("identity") or {}).get("design") or {})


def _load_source(path: Path) -> Source:
    if not path.is_file():
        raise RuntimeError(f"prior acquisition shard does not exist: {path}")
    manifest = load_manifest(path)
    if manifest is None:
        raise RuntimeError(f"prior shard has no manifest: {path}.manifest.json")
    rows, _ = canonical_rows(read_rows(path))
    errors = [row for row in rows if row.get("error")]
    if errors:
        raise RuntimeError(f"{path} has {len(errors)} unresolved error cell(s)")
    if not rows:
        raise RuntimeError(f"{path} has no successful cells")
    run_id = manifest.get("run_id")
    wrong_run = [row for row in rows if row.get("run_id") != run_id]
    if wrong_run:
        raise RuntimeError(f"{path} contains {len(wrong_run)} row(s) from another run")
    return Source(
        path=path,
        manifest=manifest,
        rows=tuple(rows),
        coordinates=frozenset(row_coordinate(row) for row in rows),
    )


def _resolve_factors(source: Source, persona_types, prompt_types):
    """The facets and framings a shard was collected under.

    A profile that leaves `persona_types` unset means "every facet", and a
    manifest written under that profile records `null`. Comparing the caller's
    `None` against a list derived from the rows would then never match, so both
    sides are resolved to concrete lists before they are compared.
    """
    design = _manifest_design(source.manifest)
    recorded_personas = design.get("persona_types") or _ordered_unique(
        row.get("persona_type")
        for row in source.rows
        if row.get("persona_id") != NO_PERSONA
    )
    recorded_prompts = design.get("prompt_types") or _ordered_unique(
        row.get("prompt_type") for row in source.rows
    )
    resolved_personas = (
        list(persona_types) if persona_types is not None else list(recorded_personas)
    )
    resolved_prompts = (
        list(prompt_types) if prompt_types is not None else list(recorded_prompts)
    )
    for label, requested, recorded in (
        ("persona_types", resolved_personas, list(recorded_personas)),
        ("prompt_types", resolved_prompts, list(recorded_prompts)),
    ):
        if requested != recorded:
            raise RuntimeError(
                f"extension {label}={requested!r} does not match "
                f"{source.path.name} {recorded!r}"
            )
    return resolved_personas, resolved_prompts


def plan_extension(
    base_output,
    personas: list,
    prompts: list,
    *,
    persona_types: list | None,
    prompt_types: list | None,
    additional_personas: int | None,
    additional_prompts: int | None,
    include_no_persona: bool,
    n_reps: int,
    seed: int,
    target_persona_ids: list[str] | None = None,
    target_prompt_ids: list[str] | None = None,
) -> ExtensionPlan:
    """Validate the prior shards and return only the cells the target still lacks.

    `base_output` is one path or several. With a frozen target the shards need
    only be error-free and inside the target; they do not have to be complete,
    because anything they are missing is simply part of what this wave runs.
    """
    paths = _as_paths(base_output)
    if not paths:
        raise RuntimeError("an extension needs at least one prior acquisition shard")
    sources = [_load_source(path) for path in paths]

    resolved_personas, resolved_prompts = _resolve_factors(
        sources[0], persona_types, prompt_types
    )
    for source in sources[1:]:
        _resolve_factors(source, resolved_personas, resolved_prompts)

    design = _manifest_design(sources[0].manifest)
    for key, requested in (("n_reps", n_reps), ("include_control", include_no_persona)):
        recorded = design.get(key)
        if recorded is not None and recorded != requested:
            raise RuntimeError(
                f"extension {key}={requested!r} does not match base {recorded!r}"
            )

    covered_rows = [row for source in sources for row in source.rows]
    covered: set[Coordinate] = set()
    for source in sources:
        _require_settled(
            source,
            personas,
            prompts,
            persona_types=resolved_personas,
            prompt_types=resolved_prompts,
            include_no_persona=include_no_persona,
            n_reps=n_reps,
        )
        overlap = covered & set(source.coordinates)
        if overlap:
            raise RuntimeError(
                f"{source.path} repeats {len(overlap)} coordinate(s) already "
                "collected by an earlier source; shards must be disjoint"
            )
        covered |= set(source.coordinates)

    base_persona_ids = _ordered_unique(
        row.get("persona_id")
        for row in covered_rows
        if row.get("persona_id") != NO_PERSONA
    )
    base_prompt_ids = _ordered_unique(row.get("prompt_id") for row in covered_rows)

    if (target_persona_ids is None) != (target_prompt_ids is None):
        raise RuntimeError("frozen targets must provide both persona and prompt IDs")

    if target_persona_ids is not None:
        all_persona_ids = _ordered_unique(target_persona_ids)
        all_prompt_ids = _ordered_unique(target_prompt_ids)
        missing_people = sorted(set(base_persona_ids) - set(all_persona_ids))
        missing_prompts = sorted(set(base_prompt_ids) - set(all_prompt_ids))
        if missing_people or missing_prompts:
            raise RuntimeError(
                "frozen extension target does not contain everything already "
                f"collected: people={missing_people}, prompts={missing_prompts}"
            )
        added_persona_ids = [
            value for value in all_persona_ids if value not in base_persona_ids
        ]
        added_prompt_ids = [
            value for value in all_prompt_ids if value not in base_prompt_ids
        ]
        if (
            additional_personas is not None
            and len(added_persona_ids) != additional_personas
        ):
            raise RuntimeError(
                f"profile requests {additional_personas} additional people, but "
                f"the frozen design adds {len(added_persona_ids)}"
            )
        if (
            additional_prompts is not None
            and len(added_prompt_ids) != additional_prompts
        ):
            raise RuntimeError(
                f"profile requests {additional_prompts} additional dilemmas, but "
                f"the frozen design adds {len(added_prompt_ids)}"
            )
    else:
        if len(sources) > 1:
            raise RuntimeError(
                "extending several shards at once needs a frozen design "
                "(--design); additional counts are only defined against one base"
            )
        if additional_personas is None or additional_prompts is None:
            raise RuntimeError(
                "an extension needs either a frozen design or additional counts"
            )
        eligible_people, eligible_prompts = eligible_design_ids(
            personas,
            prompts,
            persona_types=resolved_personas,
            prompt_types=resolved_prompts,
        )
        remaining_people = [
            value for value in eligible_people if value not in base_persona_ids
        ]
        remaining_prompts = [
            value for value in eligible_prompts if value not in base_prompt_ids
        ]
        added_persona_ids = stable_sample(
            remaining_people, additional_personas, seed, "persona-extension"
        )
        added_prompt_ids = stable_sample(
            remaining_prompts, additional_prompts, seed, "prompt-extension"
        )
        if len(added_persona_ids) != additional_personas:
            raise RuntimeError(
                f"requested {additional_personas} additional people, but only "
                f"{len(remaining_people)} eligible unused people remain"
            )
        if len(added_prompt_ids) != additional_prompts:
            raise RuntimeError(
                f"requested {additional_prompts} additional dilemmas, but only "
                f"{len(remaining_prompts)} eligible unused dilemmas remain"
            )
        all_persona_ids = [*base_persona_ids, *added_persona_ids]
        all_prompt_ids = [*base_prompt_ids, *added_prompt_ids]

    union_cells = build_cells(
        personas,
        prompts,
        persona_types=resolved_personas,
        prompt_types=resolved_prompts,
        persona_ids=all_persona_ids,
        prompt_ids=all_prompt_ids,
        include_no_persona=include_no_persona,
        n_reps=n_reps,
    )
    target = {cell_coordinate(cell) for cell in union_cells}
    outside = covered - target
    if outside:
        raise RuntimeError(
            f"{len(outside)} already-collected coordinate(s) fall outside the "
            "target design; the prior shards and this target disagree"
        )
    extension_cells = [
        cell for cell in union_cells if cell_coordinate(cell) not in covered
    ]

    return ExtensionPlan(
        sources=tuple(sources),
        base_rows=tuple(covered_rows),
        base_coordinates=frozenset(covered),
        base_persona_ids=tuple(base_persona_ids),
        base_prompt_ids=tuple(base_prompt_ids),
        added_persona_ids=tuple(added_persona_ids),
        added_prompt_ids=tuple(added_prompt_ids),
        all_persona_ids=tuple(all_persona_ids),
        all_prompt_ids=tuple(all_prompt_ids),
        target_coordinates=frozenset(target),
        cells=tuple(extension_cells),
    )


def _declared_cells(manifest: dict) -> int | None:
    """How many cells a shard's own manifest says it set out to administer."""
    design = _manifest_design(manifest)
    extension = design.get("extension") or {}
    for value in (extension.get("extension_cells"), design.get("cells")):
        if isinstance(value, int):
            return value
    return None


def _require_settled(
    source: Source,
    personas,
    prompts,
    *,
    persona_types,
    prompt_types,
    include_no_persona,
    n_reps,
) -> None:
    """Refuse to plan a wave against a shard that is still being written.

    A wave is the difference between the target and what is already collected,
    so reading a shard mid-flight would put the same coordinates in two files at
    once -- one appended by the running job, one by the new one. Every prior
    shard therefore has to be finished before it can be built on. Resuming an
    unfinished shard is a different operation: re-run it with its own --out.
    """
    declared = _declared_cells(source.manifest)
    if declared is not None:
        if len(source.coordinates) != declared:
            raise RuntimeError(
                f"{source.path} is still incomplete: {len(source.coordinates)}/"
                f"{declared} cells. Finish it before building the next wave on it."
            )
        return

    # Shards collected before manifests recorded a cell count. Their own
    # observed IDs are the design they administered, so a complete paired grid
    # over those IDs is the equivalent statement.
    persona_ids = _ordered_unique(
        row.get("persona_id")
        for row in source.rows
        if row.get("persona_id") != NO_PERSONA
    )
    prompt_ids = _ordered_unique(row.get("prompt_id") for row in source.rows)
    expected = {
        cell_coordinate(cell)
        for cell in build_cells(
            personas,
            prompts,
            persona_types=persona_types,
            prompt_types=prompt_types,
            persona_ids=persona_ids,
            prompt_ids=prompt_ids,
            include_no_persona=include_no_persona,
            n_reps=n_reps,
        )
    }
    actual = set(source.coordinates)
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        raise RuntimeError(
            f"{source.path} is not a settled paired grid: missing={len(missing)}, "
            f"unexpected={len(extra)}. Finish it before building the next wave on it."
        )


def validate_compatible_run(base_manifest: dict, extension_manifest: dict) -> None:
    """Require the same model, instrument, and source data as the prior shard.

    The acquisition digest is deliberately not compared: waves are collected
    weeks apart, and the runner proves prompt compatibility directly by
    re-deriving every prior row's stored ``prompt_digest``.
    """
    base = base_manifest.get("identity") or {}
    extension = extension_manifest.get("identity") or {}
    sections = ["model", "instrument", "data"]
    if (extension.get("model") or {}).get("backend") == "mock":
        # A --dry-run generates nothing, so it cannot contaminate a shard. Its
        # mock backend nulls temperature and top_p, which would otherwise make
        # the offline smoke test unable to exercise any extension profile.
        sections.remove("model")
    for section in sections:
        if base.get(section) != extension.get(section):
            raise RuntimeError(
                f"extension {section} configuration does not match its base run"
            )


def _successful_rows(path: Path, manifest: dict) -> list[dict]:
    rows, _ = canonical_rows(read_rows(path))
    errors = [row for row in rows if row.get("error")]
    if errors:
        raise RuntimeError(f"{path} has {len(errors)} unresolved error cell(s)")
    wrong_run = [row for row in rows if row.get("run_id") != manifest.get("run_id")]
    if wrong_run:
        raise RuntimeError(f"{path} contains {len(wrong_run)} row(s) from another run")
    return rows


def _expected_union_coordinates(rows: list[dict]) -> set[Coordinate]:
    persona_types = {
        str(row["persona_type"])
        for row in rows
        if str(row.get("persona_id")) != NO_PERSONA
    }
    persona_ids = {
        str(row["persona_id"])
        for row in rows
        if str(row.get("persona_id")) != NO_PERSONA
    }
    prompt_ids = {str(row["prompt_id"]) for row in rows}
    prompt_types = {str(row["prompt_type"]) for row in rows}
    reps = {int(row.get("rep", 0)) for row in rows}
    include_control = any(str(row.get("persona_id")) == NO_PERSONA for row in rows)

    expected = set()
    for prompt_id in prompt_ids:
        for prompt_type in prompt_types:
            for rep in reps:
                if include_control:
                    expected.add((NO_PERSONA, NO_PERSONA, prompt_type, prompt_id, rep))
                for persona_id in persona_ids:
                    for persona_type in persona_types:
                        expected.add(
                            (persona_type, persona_id, prompt_type, prompt_id, rep)
                        )
    return expected


def collect(shards, target, *, design_lock=None, probe: str | None = None) -> Path:
    """Join every acquisition shard of a study into one analysis-ready file.

    Source shards are never touched. Rows in the collection receive one new run
    ID so existing parsers and summaries deliberately pool the waves; their
    original run and cell IDs remain as audit columns.

    When `design_lock` is given the union is checked against that design's
    recorded coordinate digest, which is a far stronger guarantee than the
    self-consistency check used without one.
    """
    paths = _as_paths(shards)
    if len(paths) < 2:
        raise RuntimeError("a collection needs at least two acquisition shards")
    target_path = Path(target).resolve()
    if target_path in set(paths):
        raise RuntimeError("collection target must differ from every acquisition shard")

    manifests = []
    for path in paths:
        manifest = load_manifest(path)
        if manifest is None:
            raise RuntimeError(f"shard has no adjacent manifest: {path}")
        manifests.append(manifest)
    for manifest in manifests[1:]:
        validate_compatible_run(manifests[0], manifest)

    rows: list[dict] = []
    actual: set[Coordinate] = set()
    for path, manifest in zip(paths, manifests):
        shard_rows = _successful_rows(path, manifest)
        shard_coordinates = {row_coordinate(row) for row in shard_rows}
        overlap = actual & shard_coordinates
        if overlap:
            raise RuntimeError(
                f"{path} overlaps an earlier shard on {len(overlap)} coordinate(s)"
            )
        expected_cells = (
            (manifest.get("identity") or {}).get("design", {}).get("extension", {}) or {}
        ).get("extension_cells")
        if expected_cells is not None and len(shard_rows) != expected_cells:
            raise RuntimeError(
                f"{path} is incomplete: {len(shard_rows)}/{expected_cells} "
                "successful cells"
            )
        rows.extend(shard_rows)
        actual |= shard_coordinates

    if design_lock is not None:
        _require_design_coverage(
            design_lock,
            actual,
            manifests,
            probe=probe or str(rows[0].get("probe") or ""),
        )
    else:
        expected = _expected_union_coordinates(rows)
        if actual != expected:
            raise RuntimeError(
                f"combined collection is not fully crossed: "
                f"missing={len(expected - actual)}, "
                f"unexpected={len(actual - expected)}"
            )

    return _write_collection(rows, paths, manifests, target_path)


def _require_design_coverage(design_lock, actual, manifests, *, probe: str) -> None:
    """Check a finished collection against the design it was supposed to fill.

    Without a lock, `_expected_union_coordinates` can only ask whether the rows
    are self-consistent -- a facet missing from every wave alike would pass. The
    lock states the answer independently, so this is the check worth having.

    Data identity is compared against the shards' own manifests rather than
    re-hashing the source files: that proves the waves were *collected* against
    this design, which is the claim being made, and keeps collection working
    without the multi-gigabyte inputs to hand.
    """
    from syco.design import coordinate_digest, load_design

    lock = load_design(Path(design_lock))
    identity = lock["identity"]
    instruments = {
        str(item.get("probe")): item for item in identity.get("instruments", [])
    }
    if probe and probe not in instruments:
        raise RuntimeError(
            f"frozen design has no {probe!r} instrument; "
            f"available: {sorted(instruments)}"
        )
    for manifest in manifests:
        recorded = (manifest.get("identity") or {}).get("data") or {}
        for key in ("personas_sha256", "prompts_sha256"):
            if recorded.get(key) and recorded[key] != identity["data"][key]:
                raise RuntimeError(
                    f"a shard was collected against different {key}; it does not "
                    "belong to this design"
                )
    expected = identity.get("expected_coordinates")
    if expected is not None and len(actual) != expected:
        raise RuntimeError(
            f"collection has {len(actual)} coordinate(s); the frozen design "
            f"expects {expected}"
        )
    recorded_digest = identity.get("coordinates_sha256")
    if recorded_digest and coordinate_digest(actual) != recorded_digest:
        raise RuntimeError(
            "collection coordinates do not match the frozen design digest"
        )


def _write_collection(rows, paths, manifests, target_path: Path) -> Path:
    base_identity = manifests[0].get("identity") or {}
    last_extension = {}
    for manifest in reversed(manifests):
        extension = (
            (manifest.get("identity") or {}).get("design", {}).get("extension") or {}
        )
        if extension:
            last_extension = extension
            break
    persona_ids = _ordered_unique(
        row.get("persona_id") for row in rows if row.get("persona_id") != NO_PERSONA
    )
    prompt_ids = _ordered_unique(row.get("prompt_id") for row in rows)
    collection_identity = {
        "kind": "extension-collection",
        "model": base_identity.get("model"),
        "instrument": base_identity.get("instrument"),
        "data": base_identity.get("data"),
        "design": {
            "persona_types": (base_identity.get("design") or {}).get("persona_types"),
            "prompt_types": (base_identity.get("design") or {}).get("prompt_types"),
            "n_personas": last_extension.get("total_persona_ids") or len(persona_ids),
            "n_prompts": last_extension.get("total_prompt_ids") or len(prompt_ids),
            "n_reps": (base_identity.get("design") or {}).get("n_reps"),
            "include_control": (base_identity.get("design") or {}).get(
                "include_control"
            ),
            "persona_ids": persona_ids,
            "prompt_ids": prompt_ids,
        },
        "inputs": [
            {"path": str(path), "run_id": manifest.get("run_id")}
            for path, manifest in zip(paths, manifests)
        ],
    }
    canonical = json.dumps(collection_identity, sort_keys=True, separators=(",", ":"))
    collection_id = hashlib.sha256(canonical.encode()).hexdigest()[:20]
    collection_manifest = {
        "schema_version": 1,
        "kind": "extension-collection",
        "run_id": collection_id,
        "identity": collection_identity,
        "cells": len(rows),
    }

    target_path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_manifest(target_path)
    if target_path.exists() or existing is not None:
        if existing and existing.get("run_id") == collection_id:
            existing_rows = _successful_rows(target_path, existing)
            if len(existing_rows) == len(rows):
                return target_path
        raise RuntimeError(f"collection target already exists: {target_path}")

    model_alias = str((base_identity.get("model") or {}).get("alias") or "")
    probe = str(rows[0].get("probe") or "")
    fd, temporary = tempfile.mkstemp(
        prefix=f".{target_path.name}.", dir=target_path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for source in rows:
                row = dict(source)
                row["source_run_id"] = row.get("run_id")
                row["source_cell_key"] = row.get("cell_key")
                row["run_id"] = collection_id
                coordinate = row_coordinate(row)
                row["cell_key"] = "|".join(
                    str(value)
                    for value in (collection_id, model_alias, probe, *coordinate)
                )
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target_path)
        write_manifest(target_path, collection_manifest)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        target_path.unlink(missing_ok=True)
        raise
    return target_path


def combine_extension(base_output, extension_output, target) -> Path:
    """Two-shard form of :func:`collect`, kept for existing callers."""
    return collect([base_output, extension_output], target)
