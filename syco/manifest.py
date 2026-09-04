"""Run manifests: immutable experiment identity beside every JSONL output.

Identity answers "may these rows be pooled?", and nothing else. It therefore
records *what was administered* -- the model, the instrument, the source data,
and the exact coordinates -- plus a digest of the code that decides what bytes
reach the model.

It deliberately does NOT hash the whole repository. A study that runs for weeks
and grows by design will edit analysis code, add profiles, and reformat modules
while acquisition is in flight; hashing all of that made every such edit change
`run_id`, and a changed `run_id` breaks resume for a run that is already half
collected. The full-tree digest is still recorded, outside `identity`, as
provenance.

When an output already carries a manifest, `reconcile_manifest` adopts that
manifest's `run_id` rather than demanding the current configuration reproduce
it. Adoption is allowed only when the invariants below still hold; the runner
additionally re-derives the stored `prompt_digest` of rows already collected,
which proves prompt construction is unchanged far more directly than any hash
over source files.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

from syco import paths

MANIFEST_SCHEMA_VERSION = 2

# The files that decide which bytes reach the model, or how a completion is
# produced from them. An edit here can change an observation, so it changes
# `run_id` for a new output. Everything else in the repository -- analysis,
# parsing, orchestration, figures, profiles, tests -- cannot, and is recorded
# as `repo_digest` provenance instead.
ACQUISITION_SOURCES = (
    "syco/data.py",
    "syco/models.py",
    "syco/model_registry.py",
    "syco/prompts.py",
    "scripts/run_assumptions.py",
    "config/models.yaml",
)

# Identity fields that must match for two sets of rows to belong to the same
# run. Sampling knobs (`seed`, `n_personas`, `n_prompts`) are absent on purpose:
# the coordinates they produced are recorded explicitly, so the coordinates are
# what gets compared.
DESIGN_INVARIANTS = (
    "persona_types",
    "prompt_types",
    "n_reps",
    "include_control",
    "coordinates_sha256",
)


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest_files(relatives) -> str:
    digest = hashlib.sha256()
    for relative in sorted(relatives):
        path = paths.ROOT / relative
        digest.update(str(relative).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes() if path.is_file() else b"")
        digest.update(b"\0")
    return digest.hexdigest()


def acquisition_digest() -> str:
    """Digest of the code that determines an observation."""
    return _digest_files(ACQUISITION_SOURCES)


def repo_digest() -> str:
    """Digest of the whole project source tree. Provenance only."""
    roots = (paths.ROOT / "syco", paths.ROOT / "scripts", paths.ROOT / "config")
    files = []
    for root in roots:
        if root.is_dir():
            files.extend(p for p in root.rglob("*") if p.is_file())
    keep = [
        path.relative_to(paths.ROOT)
        for path in files
        if "__pycache__" not in path.parts and path.suffix not in {".pyc", ".pyo"}
    ]
    return _digest_files(keep)


def coordinates_digest(coordinates) -> str:
    """Content address of the exact cells a run administers."""
    canonical = json.dumps(
        sorted(tuple(str(part) for part in value) for value in coordinates),
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def _git_state() -> dict:
    def git(*args):
        try:
            return subprocess.run(
                ["git", *args], cwd=paths.ROOT, check=True, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


def manifest_path(output) -> Path:
    return Path(f"{output}.manifest.json")


def build_manifest(
    *,
    args,
    spec,
    probe,
    resolved_file=None,
    extension=None,
    frozen_design=None,
    coordinates=None,
) -> dict:
    persona_path = paths.PERSONA_PATH.resolve()
    prompt_path = paths.PROMPT_PATH.resolve()
    design = {
        "persona_types": args.persona_types,
        "prompt_types": args.prompt_types,
        "n_personas": args.n_personas,
        "n_prompts": args.n_prompts,
        "n_reps": args.n_reps,
        "include_control": not args.no_control,
        "seed": args.seed if frozen_design is None else None,
        "coordinates_sha256": (
            coordinates_digest(coordinates) if coordinates is not None else None
        ),
        "cells": len(coordinates) if coordinates is not None else None,
    }
    if frozen_design is not None:
        design["frozen_design_id"] = frozen_design["design_id"]
    if extension is not None:
        design["extension"] = extension
    identity = {
        "model": {
            "alias": spec.alias,
            "ref": spec.ref,
            "family": spec.family,
            "generation": spec.generation,
            "backend": "mock" if args.dry_run else spec.backend,
            "quantization": spec.quantization.label,
            "runtime": spec.runtime,
            "temperature": spec.provenance()["temperature"],
            "top_p": spec.provenance()["top_p"],
            "max_output_tokens": spec.max_output_tokens,
            "batch_size": spec.batch_size,
            "max_workers": spec.max_workers,
        },
        "instrument": {
            "probe": probe.kind,
            "family": probe.family,
            "n_models": probe.n_models if probe.family == "open-ended" else None,
            "dimensions": list(probe.dimensions),
            "system": args.system,
            "thinking": bool(args.thinking),
            "prompt_version": (
                getattr(args, "four_dims_prompt_version", None)
                if probe.kind == "4dims"
                else None
            ),
        },
        "design": design,
        "data": {
            "personas": str(persona_path),
            "personas_sha256": _sha256_file(persona_path),
            "prompts": str(prompt_path),
            "prompts_sha256": _sha256_file(prompt_path),
        },
        "acquisition_digest": acquisition_digest(),
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    run_id = hashlib.sha256(canonical.encode()).hexdigest()[:20]
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "identity": identity,
        "artifact": {
            "quantized_file": resolved_file or spec.quantization.resolved_file,
            "design_file": (
                str(args.design.resolve()) if getattr(args, "design", None) else None
            ),
        },
        "provenance": {"repo_digest": repo_digest()},
        "git": _git_state(),
    }


def load_manifest(output) -> dict | None:
    path = manifest_path(output)
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_manifest(output, manifest: dict) -> Path:
    target = manifest_path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return target


def _section(manifest: dict, name: str) -> dict:
    return dict((manifest.get("identity") or {}).get(name) or {})


COMPARED_FIELDS = {
    "model": (
        "alias",
        "ref",
        "family",
        "generation",
        "backend",
        "quantization",
        "runtime",
        "temperature",
        "top_p",
        "max_output_tokens",
    ),
    "instrument": (
        "probe",
        "family",
        "n_models",
        "dimensions",
        "system",
        "thinking",
        "prompt_version",
    ),
    "data": ("personas_sha256", "prompts_sha256"),
    "design": DESIGN_INVARIANTS,
}


def identity_conflicts(
    existing: dict, expected: dict, *, sections=None
) -> list[str]:
    """Reasons two sets of rows may not be pooled.

    A field absent from one side is not a conflict: manifests written before
    schema 2 have no `coordinates_sha256`, and refusing those would strand every
    output collected so far. That tolerance is also what lets a study started
    under one schema finish under the next.

    `sections` narrows the comparison. Merging across models drops `model`,
    since differing there is the entire point of the merge.
    """
    conflicts = []
    for section in sections or COMPARED_FIELDS:
        fields = COMPARED_FIELDS[section]
        old = _section(existing, section)
        new = _section(expected, section)
        for field in fields:
            if field not in old or field not in new:
                continue
            if old[field] is None or new[field] is None:
                continue
            if old[field] != new[field]:
                conflicts.append(
                    f"{section}.{field}: recorded {old[field]!r}, now {new[field]!r}"
                )
    return conflicts


def _revision(expected: dict) -> dict:
    identity = expected.get("identity") or {}
    return {
        "at": dt.datetime.now(dt.UTC).isoformat(),
        "run_id_if_new": expected.get("run_id"),
        "acquisition_digest": identity.get("acquisition_digest"),
        "repo_digest": (expected.get("provenance") or {}).get("repo_digest"),
        "git": expected.get("git"),
    }


def reconcile_manifest(output, expected: dict, *, has_output: bool, write: bool = True):
    """Return the manifest that governs `output`, and whether it was adopted.

    A new output takes `expected`. An output that already has a manifest keeps
    its own `run_id` -- that identifier is embedded in every `cell_key` already
    written, so adopting it is what lets a long run resume across code changes.
    Adoption is refused when `identity_conflicts` finds a difference that would
    make the old and new rows different observations.
    """
    existing = load_manifest(output)
    if existing is None:
        if has_output:
            raise RuntimeError(
                f"{output} already contains rows but has no run manifest. Choose a "
                "new --out; legacy output cannot be resumed safely."
            )
        if write:
            write_manifest(output, expected)
        return expected, False

    conflicts = identity_conflicts(existing, expected)
    if conflicts:
        raise RuntimeError(
            f"{output} belongs to run {existing.get('run_id')} and the current "
            "configuration would collect different observations:\n  - "
            + "\n  - ".join(conflicts)
            + "\nChoose a new --out, or restore the configuration this output was "
            "collected under."
        )

    if existing.get("run_id") == expected.get("run_id"):
        return existing, False

    # Same experiment, different code or sampling arguments. Keep the recorded
    # identity authoritative and append what changed, so the drift is visible
    # in the artifact rather than silently accepted.
    adopted = json.loads(json.dumps(existing))
    revisions = list(adopted.get("revisions") or [])
    revision = _revision(expected)
    if not revisions or revisions[-1].get("run_id_if_new") != revision["run_id_if_new"]:
        revisions.append(revision)
    adopted["revisions"] = revisions
    if write:
        write_manifest(output, adopted)
    return adopted, True


def ensure_manifest(output, expected: dict, *, has_output: bool) -> Path:
    """Backwards-compatible wrapper: reconcile, then return the manifest path."""
    reconcile_manifest(output, expected, has_output=has_output)
    return manifest_path(output)
