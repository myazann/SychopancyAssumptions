"""Immutable, model-independent study designs used by acquisition itself."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

CONTROL_ID = "none"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def portable_path(path: Path) -> str:
    """Prefer repository-relative provenance without hiding external inputs."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def ordered_unique(values: Iterable[Any]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values))


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")  # noqa: TRY004
    return value


def read_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSONL at {path}:{number}: {exc}") from exc
            if isinstance(value, dict):
                rows.append(value)
    return rows


def row_coordinate(row: dict) -> tuple[str, str, str, str, int]:
    return (
        str(row.get("persona_type")),
        str(row.get("persona_id")),
        str(row.get("prompt_type")),
        str(row.get("prompt_id")),
        int(row.get("rep", 0)),
    )


def coordinate_digest(coordinates: Iterable[tuple]) -> str:
    return digest(sorted(coordinates))


def target_coordinates(
    *,
    persona_ids: list[str],
    persona_types: list[str],
    prompt_ids: list[str],
    prompt_types: list[str],
    n_reps: int,
    include_control: bool,
    control_type: str,
    control_id: str,
) -> set[tuple[str, str, str, str, int]]:
    coordinates: set[tuple[str, str, str, str, int]] = set()
    for prompt_id in prompt_ids:
        for prompt_type in prompt_types:
            for rep in range(n_reps):
                if include_control:
                    coordinates.add(
                        (control_type, control_id, prompt_type, prompt_id, rep)
                    )
                for persona_id in persona_ids:
                    for persona_type in persona_types:
                        coordinates.add(
                            (persona_type, persona_id, prompt_type, prompt_id, rep)
                        )
    return coordinates


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def extension_identity(manifest: dict, path: Path) -> dict:
    extension = (
        (manifest.get("identity") or {})
        .get("design", {})
        .get("extension")
    )
    if not isinstance(extension, dict):
        raise RuntimeError(  # noqa: TRY004
            f"manifest is not an extension run: {path}"
        )
    return extension


def make_design_lock(name: str, manifest_paths: list[Path]) -> dict:
    if not manifest_paths:
        raise RuntimeError("at least one extension manifest is required")

    records = [(path.resolve(), read_json(path.resolve())) for path in manifest_paths]
    reference_path, reference = records[0]
    reference_identity = reference.get("identity") or {}
    reference_design = reference_identity.get("design") or {}
    reference_extension = extension_identity(reference, reference_path)

    common_extension_keys = (
        "base_persona_ids",
        "base_prompt_ids",
        "added_persona_ids",
        "added_prompt_ids",
        "total_persona_ids",
        "total_prompt_ids",
    )
    for path, manifest in records[1:]:
        identity = manifest.get("identity") or {}
        design = identity.get("design") or {}
        extension = extension_identity(manifest, path)
        if identity.get("data") != reference_identity.get("data"):
            raise RuntimeError(f"data identity differs between manifests: {path}")
        for key in ("prompt_types", "n_reps", "include_control", "seed"):
            if design.get(key) != reference_design.get(key):
                raise RuntimeError(f"design field {key!r} differs: {path}")
        for key in common_extension_keys:
            if extension.get(key) != reference_extension.get(key):
                raise RuntimeError(f"extension field {key!r} differs: {path}")

    base_outputs = [
        Path(extension_identity(manifest, path)["base_output"]).resolve()
        for path, manifest in records
    ]
    base_rows = read_rows(base_outputs[0])
    base_run_id = extension_identity(reference, reference_path).get("base_run_id")
    base_rows = [row for row in base_rows if row.get("run_id") == base_run_id]
    if not base_rows:
        raise RuntimeError(f"no rows for base run {base_run_id}: {base_outputs[0]}")

    persona_types = ordered_unique(
        row["persona_type"]
        for row in base_rows
        if str(row.get("persona_id")) != CONTROL_ID
    )
    prompt_types = list(reference_design.get("prompt_types") or ordered_unique(
        row["prompt_type"] for row in base_rows
    ))
    control_rows = [
        row for row in base_rows if str(row.get("persona_id")) == CONTROL_ID
    ]
    control_type = str(control_rows[0]["persona_type"]) if control_rows else CONTROL_ID
    include_control = bool(reference_design.get("include_control"))
    n_reps = int(reference_design.get("n_reps", 1))

    base_persona_ids = list(reference_extension["base_persona_ids"])
    base_prompt_ids = list(reference_extension["base_prompt_ids"])
    added_persona_ids = list(reference_extension["added_persona_ids"])
    added_prompt_ids = list(reference_extension["added_prompt_ids"])
    persona_ids = [*base_persona_ids, *added_persona_ids]
    prompt_ids = [*base_prompt_ids, *added_prompt_ids]

    if len(persona_ids) != len(set(persona_ids)):
        raise RuntimeError("base and added persona IDs overlap")
    if len(prompt_ids) != len(set(prompt_ids)):
        raise RuntimeError("base and added prompt IDs overlap")

    actual_base = {row_coordinate(row) for row in base_rows}
    expected_base = target_coordinates(
        persona_ids=base_persona_ids,
        persona_types=persona_types,
        prompt_ids=base_prompt_ids,
        prompt_types=prompt_types,
        n_reps=n_reps,
        include_control=include_control,
        control_type=control_type,
        control_id=CONTROL_ID,
    )
    if actual_base != expected_base:
        raise RuntimeError(
            "base output is not the explicit paired design: "
            f"missing={len(expected_base - actual_base)}, "
            f"unexpected={len(actual_base - expected_base)}"
        )
    recorded_digest = reference_extension.get("base_coordinates_sha256")
    if recorded_digest and coordinate_digest(actual_base) != recorded_digest:
        raise RuntimeError("base coordinate digest does not match extension manifest")

    factors = {
        "persona_types": persona_types,
        "prompt_types": prompt_types,
        "n_reps": n_reps,
        "include_control": include_control,
        "control": {"persona_type": control_type, "persona_id": CONTROL_ID},
    }
    data = {
        "personas_sha256": reference_identity["data"]["personas_sha256"],
        "prompts_sha256": reference_identity["data"]["prompts_sha256"],
    }
    parent_identity = {
        "data": data,
        "selection": {
            "persona_ids": base_persona_ids,
            "prompt_ids": base_prompt_ids,
        },
        "factors": factors,
    }
    target = target_coordinates(
        persona_ids=persona_ids,
        persona_types=persona_types,
        prompt_ids=prompt_ids,
        prompt_types=prompt_types,
        n_reps=n_reps,
        include_control=include_control,
        control_type=control_type,
        control_id=CONTROL_ID,
    )
    instruments = []
    for path, manifest in records:
        instrument = dict((manifest.get("identity") or {}).get("instrument") or {})
        instrument["expected_cells"] = len(target)
        instruments.append(instrument)
    instruments.sort(key=lambda item: str(item.get("probe")))

    identity = {
        "name": name,
        "data": data,
        "selection": {
            "persona_ids": persona_ids,
            "prompt_ids": prompt_ids,
        },
        "factors": factors,
        "extension": {
            "mode": "full-cross",
            "parent_design_id": digest(parent_identity),
            "base_persona_ids": base_persona_ids,
            "base_prompt_ids": base_prompt_ids,
            "added_persona_ids": added_persona_ids,
            "added_prompt_ids": added_prompt_ids,
        },
        "expected_coordinates": len(target),
        "coordinates_sha256": coordinate_digest(target),
        "instruments": instruments,
    }
    return {
        "schema_version": 1,
        "kind": "syco-design-lock",
        "design_id": digest(identity),
        "identity": identity,
        "provenance": {
            "source_manifests": [portable_path(path) for path, _ in records],
            "base_outputs": [portable_path(path) for path in base_outputs],
            "source_runs": [
                {
                    "run_id": manifest.get("run_id"),
                    "probe": ((manifest.get("identity") or {}).get("instrument") or {}).get("probe"),
                }
                for _, manifest in records
            ],
            "source_digests": sorted({
                str(
                    (manifest.get("identity") or {}).get("acquisition_digest")
                    or (manifest.get("identity") or {}).get("source_digest")
                )
                for _, manifest in records
            }),
            "selection_seed": reference_design.get("seed"),
        },
    }


def freeze_design_lock(name: str, output_paths: list[Path]) -> dict:
    """Lock the design a plain run already administered, so waves can start.

    `make_design_lock` reconstructs a target from *extension* manifests, which
    only helps once an extension exists. This starts the chain instead: give it
    the finished acquisition outputs -- one per instrument -- and it records the
    people, dilemmas, and factors they actually cover.

    The resulting design ID is not the `extension.parent_design_id` that
    `create` writes. That field is a digest of a reconstructed identity holding
    only data, selection, and factors; a real lock also carries a name,
    coordinate digest, and instruments. Both identify the same design, but only
    this one is a file you can verify and extend.
    """
    if not output_paths:
        raise RuntimeError("at least one acquisition output is required")

    records = []
    for path in output_paths:
        resolved = path.resolve(strict=True)
        manifest_file = Path(f"{resolved}.manifest.json")
        if not manifest_file.is_file():
            raise RuntimeError(f"output has no manifest: {manifest_file}")
        manifest = read_json(manifest_file)
        rows = read_rows(resolved)
        run_id = manifest.get("run_id")
        rows = [row for row in rows if row.get("run_id") == run_id]
        if not rows:
            raise RuntimeError(f"no rows for run {run_id}: {resolved}")
        errors = [row for row in rows if row.get("error")]
        if errors:
            raise RuntimeError(f"{resolved} has {len(errors)} unresolved error cell(s)")
        records.append((resolved, manifest, rows))

    reference_path, reference, reference_rows = records[0]
    reference_identity = reference.get("identity") or {}
    reference_design = reference_identity.get("design") or {}
    for path, manifest, _ in records[1:]:
        identity = manifest.get("identity") or {}
        if identity.get("data") != reference_identity.get("data"):
            raise RuntimeError(f"data identity differs between outputs: {path}")

    persona_types = list(reference_design.get("persona_types") or []) or ordered_unique(
        row["persona_type"]
        for row in reference_rows
        if str(row.get("persona_id")) != CONTROL_ID
    )
    prompt_types = list(reference_design.get("prompt_types") or []) or ordered_unique(
        row["prompt_type"] for row in reference_rows
    )
    control_rows = [
        row for row in reference_rows if str(row.get("persona_id")) == CONTROL_ID
    ]
    control_type = str(control_rows[0]["persona_type"]) if control_rows else CONTROL_ID
    include_control = bool(control_rows)
    n_reps = int(reference_design.get("n_reps", 1))
    persona_ids = ordered_unique(
        row["persona_id"]
        for row in reference_rows
        if str(row.get("persona_id")) != CONTROL_ID
    )
    prompt_ids = ordered_unique(row["prompt_id"] for row in reference_rows)

    factors = {
        "persona_types": persona_types,
        "prompt_types": prompt_types,
        "n_reps": n_reps,
        "include_control": include_control,
        "control": {"persona_type": control_type, "persona_id": CONTROL_ID},
    }
    target = target_coordinates(
        persona_ids=persona_ids,
        persona_types=persona_types,
        prompt_ids=prompt_ids,
        prompt_types=prompt_types,
        n_reps=n_reps,
        include_control=include_control,
        control_type=control_type,
        control_id=CONTROL_ID,
    )
    for path, _, rows in records:
        actual = {row_coordinate(row) for row in rows}
        if actual != target:
            raise RuntimeError(
                f"{path} is not a complete paired grid over its own IDs: "
                f"missing={len(target - actual)}, unexpected={len(actual - target)}"
            )

    instruments = []
    for _, manifest, _ in records:
        instrument = dict((manifest.get("identity") or {}).get("instrument") or {})
        instrument["expected_cells"] = len(target)
        instruments.append(instrument)
    instruments.sort(key=lambda item: str(item.get("probe")))

    identity = {
        "name": name,
        "data": {
            "personas_sha256": reference_identity["data"]["personas_sha256"],
            "prompts_sha256": reference_identity["data"]["prompts_sha256"],
        },
        "selection": {"persona_ids": persona_ids, "prompt_ids": prompt_ids},
        "factors": factors,
        "expected_coordinates": len(target),
        "coordinates_sha256": coordinate_digest(target),
        "instruments": instruments,
    }
    return {
        "schema_version": 1,
        "kind": "syco-design-lock",
        "design_id": digest(identity),
        "identity": identity,
        "provenance": {
            "frozen_from": [portable_path(path) for path, _, _ in records],
            "source_runs": [
                {
                    "run_id": manifest.get("run_id"),
                    "probe": (
                        (manifest.get("identity") or {}).get("instrument") or {}
                    ).get("probe"),
                }
                for _, manifest, _ in records
            ],
            "selection_seed": reference_design.get("seed"),
        },
    }


def verify_design_lock(path: Path) -> dict:
    lock = read_json(path)
    if lock.get("kind") != "syco-design-lock" or lock.get("schema_version") != 1:
        raise RuntimeError(f"unsupported design lock: {path}")
    identity = lock.get("identity")
    if not isinstance(identity, dict):
        raise RuntimeError("design lock has no identity object")  # noqa: TRY004
    actual = digest(identity)
    if actual != lock.get("design_id"):
        raise RuntimeError(
            f"design ID mismatch: recorded={lock.get('design_id')} actual={actual}"
        )
    selection = identity["selection"]
    factors = identity["factors"]
    target = target_coordinates(
        persona_ids=selection["persona_ids"],
        persona_types=factors["persona_types"],
        prompt_ids=selection["prompt_ids"],
        prompt_types=factors["prompt_types"],
        n_reps=int(factors["n_reps"]),
        include_control=bool(factors["include_control"]),
        control_type=factors["control"]["persona_type"],
        control_id=factors["control"]["persona_id"],
    )
    if len(target) != identity.get("expected_coordinates"):
        raise RuntimeError("expected coordinate count does not match explicit selection")
    if coordinate_digest(target) != identity.get("coordinates_sha256"):
        raise RuntimeError("coordinate digest does not match explicit selection")
    return {
        "design_id": actual,
        "personas": len(selection["persona_ids"]),
        "prompts": len(selection["prompt_ids"]),
        "coordinates_per_instrument": len(target),
        "instruments": [item.get("probe") for item in identity["instruments"]],
    }


def load_design(path: Path) -> dict:
    """Load and structurally verify a frozen study design."""
    path = path.resolve(strict=True)
    verify_design_lock(path)
    return read_json(path)


def selection_for(
    path: Path,
    *,
    probe: str,
    persona_path: Path,
    prompt_path: Path,
) -> dict:
    """Return the exact grid selection after checking data and instrument IDs."""
    lock = load_design(path)
    identity = lock["identity"]
    data = identity["data"]
    if file_digest(persona_path) != data["personas_sha256"]:
        raise RuntimeError("persona data does not match the frozen study design")
    if file_digest(prompt_path) != data["prompts_sha256"]:
        raise RuntimeError("prompt data does not match the frozen study design")
    instruments = {
        str(item.get("probe")): item for item in identity.get("instruments", [])
    }
    if probe not in instruments:
        raise RuntimeError(
            f"study design has no {probe!r} instrument; "
            f"available: {sorted(instruments)}"
        )
    return {
        "path": str(path.resolve()),
        "design_id": lock["design_id"],
        "selection": identity["selection"],
        "factors": identity["factors"],
        "instrument": instruments[probe],
        "expected_coordinates": identity.get("expected_coordinates"),
        "coordinates_sha256": identity.get("coordinates_sha256"),
    }


def extend_design_lock(
    *,
    name: str,
    parent_path: Path,
    persona_path: Path,
    prompt_path: Path,
    add_personas: int,
    add_prompts: int,
    seed: int,
) -> dict:
    """Create the next target design before any acquisition is submitted."""
    if add_personas < 1 or add_prompts < 1:
        raise RuntimeError("add-personas and add-prompts must both be positive")
    verify_design_lock(parent_path)
    parent = read_json(parent_path)
    parent_identity = parent["identity"]
    expected_data = parent_identity["data"]
    actual_personas = file_digest(persona_path)
    actual_prompts = file_digest(prompt_path)
    if actual_personas != expected_data["personas_sha256"]:
        raise RuntimeError("persona data does not match the parent design")
    if actual_prompts != expected_data["prompts_sha256"]:
        raise RuntimeError("prompt data does not match the parent design")

    try:
        from syco.data import load_personas, load_prompts
        from syco.grid import eligible_design_ids, stable_sample
    except ImportError as exc:
        raise RuntimeError(
            "extending a design requires the project's frozen environment; "
            "run this command with .venv/bin/python"
        ) from exc

    personas, _ = load_personas(persona_path)
    prompts = load_prompts(prompt_path)
    factors = parent_identity["factors"]
    eligible_personas, eligible_prompts = eligible_design_ids(
        personas,
        prompts,
        persona_types=factors["persona_types"],
        prompt_types=factors["prompt_types"],
    )
    old_personas = list(parent_identity["selection"]["persona_ids"])
    old_prompts = list(parent_identity["selection"]["prompt_ids"])
    remaining_personas = [value for value in eligible_personas if value not in old_personas]
    remaining_prompts = [value for value in eligible_prompts if value not in old_prompts]
    tag = parent["design_id"]
    added_personas = stable_sample(
        remaining_personas, add_personas, seed, f"persona-extension|{tag}"
    )
    added_prompts = stable_sample(
        remaining_prompts, add_prompts, seed, f"prompt-extension|{tag}"
    )
    if len(added_personas) != add_personas:
        raise RuntimeError(
            f"requested {add_personas} people but only {len(remaining_personas)} "
            "eligible unused IDs remain"
        )
    if len(added_prompts) != add_prompts:
        raise RuntimeError(
            f"requested {add_prompts} prompts but only {len(remaining_prompts)} "
            "eligible unused IDs remain"
        )

    identity = copy.deepcopy(parent_identity)
    identity["name"] = name
    identity["selection"] = {
        "persona_ids": [*old_personas, *added_personas],
        "prompt_ids": [*old_prompts, *added_prompts],
    }
    identity["extension"] = {
        "mode": "full-cross",
        "parent_design_id": parent["design_id"],
        "added_persona_ids": added_personas,
        "added_prompt_ids": added_prompts,
    }
    target = target_coordinates(
        persona_ids=identity["selection"]["persona_ids"],
        persona_types=factors["persona_types"],
        prompt_ids=identity["selection"]["prompt_ids"],
        prompt_types=factors["prompt_types"],
        n_reps=int(factors["n_reps"]),
        include_control=bool(factors["include_control"]),
        control_type=factors["control"]["persona_type"],
        control_id=factors["control"]["persona_id"],
    )
    identity["expected_coordinates"] = len(target)
    identity["coordinates_sha256"] = coordinate_digest(target)
    for instrument in identity["instruments"]:
        instrument["expected_cells"] = len(target)
    return {
        "schema_version": 1,
        "kind": "syco-design-lock",
        "design_id": digest(identity),
        "identity": identity,
        "provenance": {
            "parent_lock": portable_path(parent_path),
            "selection_seed": seed,
            "data_paths": {
                "personas": portable_path(persona_path),
                "prompts": portable_path(prompt_path),
            },
        },
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create", help="create a lock from extension manifests")
    create.add_argument("--name", required=True)
    create.add_argument("--manifest", action="append", required=True, type=Path)
    create.add_argument("--output", required=True, type=Path)
    freeze = subparsers.add_parser(
        "freeze", help="lock the design a finished run already administered"
    )
    freeze.add_argument("--name", required=True)
    freeze.add_argument(
        "--run",
        action="append",
        required=True,
        type=Path,
        help="a finished acquisition JSONL; repeat once per instrument",
    )
    freeze.add_argument("--output", required=True, type=Path)
    verify = subparsers.add_parser("verify", help="verify a design lock")
    verify.add_argument("path", type=Path)
    extend = subparsers.add_parser(
        "extend", help="select disjoint IDs and lock the next full-cross target"
    )
    extend.add_argument("--name", required=True)
    extend.add_argument("--from", dest="parent", required=True, type=Path)
    extend.add_argument("--personas", required=True, type=Path)
    extend.add_argument("--prompts", required=True, type=Path)
    extend.add_argument("--add-personas", required=True, type=int)
    extend.add_argument("--add-prompts", required=True, type=int)
    extend.add_argument("--seed", required=True, type=int)
    extend.add_argument("--output", required=True, type=Path)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "create":
        value = make_design_lock(args.name, args.manifest)
        atomic_json(args.output.resolve(), value)
        print(f"created {args.output.resolve()}")
        print(json.dumps(verify_design_lock(args.output.resolve()), indent=2))
        return 0
    if args.command == "freeze":
        value = freeze_design_lock(args.name, list(args.run))
        atomic_json(args.output.resolve(), value)
        print(f"created {args.output.resolve()}")
        print(json.dumps(verify_design_lock(args.output.resolve()), indent=2))
        return 0
    if args.command == "extend":
        value = extend_design_lock(
            name=args.name,
            parent_path=args.parent.resolve(strict=True),
            persona_path=args.personas.resolve(strict=True),
            prompt_path=args.prompts.resolve(strict=True),
            add_personas=args.add_personas,
            add_prompts=args.add_prompts,
            seed=args.seed,
        )
        atomic_json(args.output.resolve(), value)
        print(f"created {args.output.resolve()}")
        print(json.dumps(verify_design_lock(args.output.resolve()), indent=2))
        return 0
    value = verify_design_lock(args.path.resolve())
    print(json.dumps(value, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
