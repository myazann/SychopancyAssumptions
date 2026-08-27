"""Run manifests: immutable experiment identity beside every JSONL output."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

from syco import paths

MANIFEST_SCHEMA_VERSION = 1


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_digest() -> str:
    digest = hashlib.sha256()
    roots = (paths.ROOT / "syco", paths.ROOT / "scripts", paths.ROOT / "config")
    files = []
    for root in roots:
        if root.is_dir():
            files.extend(p for p in root.rglob("*") if p.is_file())
    for path in sorted(files):
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        digest.update(str(path.relative_to(paths.ROOT)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


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


def build_manifest(*, args, spec, probe, resolved_file=None) -> dict:
    persona_path = paths.PERSONA_PATH.resolve()
    prompt_path = paths.PROMPT_PATH.resolve()
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
            "n_models": probe.n_models,
            "system": args.system,
            "thinking": bool(args.thinking),
        },
        "design": {
            "persona_types": args.persona_types,
            "prompt_types": args.prompt_types,
            "n_personas": args.n_personas,
            "n_prompts": args.n_prompts,
            "n_reps": args.n_reps,
            "include_control": not args.no_control,
            "seed": args.seed,
        },
        "data": {
            "personas": str(persona_path),
            "personas_sha256": _sha256_file(persona_path),
            "prompts": str(prompt_path),
            "prompts_sha256": _sha256_file(prompt_path),
        },
        "source_digest": _source_digest(),
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    run_id = hashlib.sha256(canonical.encode()).hexdigest()[:20]
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "identity": identity,
        "artifact": {
            "quantized_file": resolved_file or spec.quantization.resolved_file,
        },
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


def ensure_manifest(output, expected: dict, *, has_output: bool) -> Path:
    existing = load_manifest(output)
    if existing is None:
        if has_output:
            raise RuntimeError(
                f"{output} already contains rows but has no run manifest. Choose a "
                "new --out; legacy output cannot be resumed safely."
            )
        return write_manifest(output, expected)
    if existing.get("run_id") != expected.get("run_id"):
        raise RuntimeError(
            f"{output} belongs to run {existing.get('run_id')}, but the current "
            f"configuration is run {expected.get('run_id')}. Choose a new --out."
        )
    return manifest_path(output)
