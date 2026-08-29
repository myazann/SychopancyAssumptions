"""Capture and verify self-contained snapshots of a study run.

The capture is read-only with respect to acquisition inputs and outputs. It
copies manifests and log snapshots, but never copies or opens result JSONL for
writing. A generated snapshot can be refreshed after jobs finish.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any

SOURCE_EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", ".ruff_cache", "logs"}
SOURCE_EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def command(args: list[str], cwd: Path) -> dict:
    try:
        result = subprocess.run(
            args,
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
            env={
                **os.environ,
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            },
        )
    except OSError as exc:
        return {"argv": args, "available": False, "error": str(exc)}
    return {
        "argv": args,
        "available": True,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def git_text(repo: Path, *args: str) -> str:
    result = command(["git", *args], repo)
    if not result.get("available") or result.get("returncode") != 0:
        return ""
    return str(result.get("stdout", ""))


def source_files(repo: Path) -> list[Path]:
    """Use Git's view of source so new project files are captured automatically."""
    listed = git_text(
        repo, "ls-files", "--cached", "--others", "--exclude-standard"
    ).splitlines()
    files: list[Path] = []
    for relative in listed:
        path = repo / relative
        if not path.is_file():
            continue
        inside = path.relative_to(repo)
        if SOURCE_EXCLUDED_PARTS.intersection(inside.parts):
            continue
        if path.suffix in SOURCE_EXCLUDED_SUFFIXES:
            continue
        files.append(path)
    return sorted(set(files), key=lambda path: str(path.relative_to(repo)))


def capture_source(repo: Path, bundle: Path) -> dict:
    files = source_files(repo)
    inventory = []
    for path in files:
        stat = path.stat()
        inventory.append({
            "path": str(path.relative_to(repo)),
            "size": stat.st_size,
            "mode": oct(stat.st_mode & 0o777),
            "sha256": sha256_file(path),
        })
    atomic_json(bundle / "source-files.json", inventory)

    archive = bundle / "source.tar.gz"
    with tarfile.open(archive, "w:gz", format=tarfile.PAX_FORMAT) as handle:
        for path in files:
            handle.add(path, arcname=str(path.relative_to(repo)), recursive=False)

    git = {
        "head": git_text(repo, "rev-parse", "HEAD").strip() or None,
        "branch": git_text(repo, "branch", "--show-current").strip() or None,
        "status_porcelain_v2": git_text(repo, "status", "--porcelain=v2"),
        "untracked": git_text(repo, "ls-files", "--others", "--exclude-standard").splitlines(),
    }
    atomic_json(bundle / "git.json", git)
    atomic_text(bundle / "working-tree.patch", git_text(repo, "diff", "--binary", "HEAD"))
    return {
        "archive": archive.name,
        "archive_sha256": sha256_file(archive),
        "file_inventory": "source-files.json",
        "files": len(files),
        "git": "git.json",
        "tracked_patch": "working-tree.patch",
    }


def capture_environment(repo: Path, python: Path) -> dict:
    python_info = command(
        [
            str(python),
            "-c",
            (
                "import json,platform,sys; "
                "print(json.dumps({'version':sys.version,'executable':sys.executable,"
                "'implementation':platform.python_implementation(),"
                "'platform':platform.platform(),'prefix':sys.prefix,"
                "'base_prefix':sys.base_prefix}))"
            ),
        ],
        repo,
    )
    python_value: dict = python_info
    if python_info.get("returncode") == 0:
        try:
            python_value = json.loads(str(python_info["stdout"]))
        except json.JSONDecodeError:
            pass

    distributions = command(
        [
            str(python),
            "-c",
            (
                "import importlib.metadata as m,json; rows=[]; "
                "[(rows.append({'name':(d.metadata.get('Name') or ''),"
                "'version':d.version,'direct_url':d.read_text('direct_url.json')})) "
                "for d in m.distributions()]; "
                "print(json.dumps(sorted(rows,key=lambda x:x['name'].lower())))"
            ),
        ],
        repo,
    )
    packages: list[dict] = []
    if distributions.get("returncode") == 0:
        try:
            packages = json.loads(str(distributions["stdout"]))
        except json.JSONDecodeError:
            packages = []
    freeze = command([str(python), "-m", "pip", "freeze", "--all"], repo)

    native_runtime = []
    llama_location = command(
        [
            str(python),
            "-c",
            (
                "import importlib.util,pathlib; s=importlib.util.find_spec('llama_cpp'); "
                "print(pathlib.Path(s.origin).parent if s and s.origin else '')"
            ),
        ],
        repo,
    )
    if llama_location.get("returncode") == 0:
        location_text = str(llama_location.get("stdout", "")).strip()
        location = Path(location_text) if location_text else None
        if location and location.is_dir():
            for path in sorted(location.rglob("*.so*")):
                if path.is_file():
                    native_runtime.append(artifact_record(path, hash_content=True))
    return {
        "capturing_python": sys.version,
        "experiment_python": python_value,
        "packages": packages,
        "package_inventory_command": distributions,
        "pip_freeze_command": freeze,
        "native_runtime": native_runtime,
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "runtime_environment": {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "HF_HOME",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "TOKENIZERS_PARALLELISM",
            )
        },
        "commands": {
            "uname": command(["uname", "-a"], repo),
            "lscpu": command(["lscpu"], repo),
            "nvidia_smi": command(["nvidia-smi", "-q"], repo),
            "slurm_squeue": command(["squeue", "-u", os.environ.get("USER", "")], repo),
        },
    }


def manifest_digest(manifest: dict) -> str:
    """The acquisition digest of a run, across manifest schema versions.

    Schema 1 hashed the whole source tree as `source_digest`; schema 2 hashes
    only the files that decide an observation. Both identify the acquisition
    code, so either serves here.
    """
    identity = manifest.get("identity") or {}
    return str(
        identity.get("acquisition_digest") or identity.get("source_digest") or ""
    )


def read_manifest(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"manifest is not an object: {path}")  # noqa: TRY004
    return value


def artifact_record(path: Path, *, hash_content: bool) -> dict:
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    record = {
        "requested_path": str(path),
        "resolved_path": str(resolved),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": None,
        "hash_source": None,
    }
    if hash_content:
        record["sha256"] = sha256_file(resolved)
        record["hash_source"] = "content"
    elif len(resolved.name) == 64 and all(char in "0123456789abcdef" for char in resolved.name):
        record["sha256"] = resolved.name
        record["hash_source"] = "huggingface-cache-blob-name-unverified"
    return record


def copy_evidence(paths: list[Path], target: Path) -> list[dict]:
    target.mkdir(parents=True, exist_ok=True)
    records = []
    used: set[str] = set()
    for source in paths:
        source = source.resolve(strict=True)
        name = source.name
        if name in used:
            name = f"{hashlib.sha256(str(source).encode()).hexdigest()[:10]}-{name}"
        used.add(name)
        destination = target / name
        shutil.copy2(source, destination)
        records.append({
            "source": str(source),
            "snapshot": str(destination.relative_to(target.parent)),
            "size": destination.stat().st_size,
            "sha256": sha256_file(destination),
        })
    return records


def write_checksums(bundle: Path) -> None:
    entries = []
    for path in sorted(bundle.rglob("*")):
        if not path.is_file() or path.name == "checksums.sha256":
            continue
        entries.append(f"{sha256_file(path)}  {path.relative_to(bundle)}")
    atomic_text(bundle / "checksums.sha256", "\n".join(entries) + "\n")


def create_parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--name", required=True)
    result.add_argument("--repo", type=Path, default=Path.cwd())
    result.add_argument("--output-root", type=Path, default=Path("results/snapshots"))
    result.add_argument("--python", type=Path, default=Path(".venv/bin/python"))
    result.add_argument("--manifest", action="append", type=Path, required=True)
    result.add_argument("--design", dest="design_lock", type=Path)
    result.add_argument("--model-artifact", action="append", type=Path, default=[])
    result.add_argument("--tokenizer-file", action="append", type=Path, default=[])
    result.add_argument("--evidence", action="append", type=Path, default=[])
    result.add_argument("--job", action="append", default=[], help="free-form immutable job relation")
    result.add_argument("--hash-large-artifacts", action="store_true")
    return result


def capture(args) -> Path:
    repo = args.repo.resolve(strict=True)
    output_root = (repo / args.output_root).resolve() if not args.output_root.is_absolute() else args.output_root.resolve()
    manifests = [(repo / value).resolve() if not value.is_absolute() else value.resolve() for value in args.manifest]
    values = [read_manifest(path) for path in manifests]
    # A study collected in waves legitimately spans several acquisition
    # digests: shards are immutable, and a later wave may run under edited
    # code. Record every digest present rather than demanding one.
    source_digests = sorted({manifest_digest(value) for value in values})
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    tag = (
        source_digests[0][:12]
        if len(source_digests) == 1
        else hashlib.sha256("|".join(source_digests).encode()).hexdigest()[:12]
    )
    bundle = output_root / f"{args.name}-{stamp}-{tag}"
    bundle.mkdir(parents=True, exist_ok=False)

    source = capture_source(repo, bundle)
    python = args.python if args.python.is_absolute() else repo / args.python
    if not python.is_file():
        raise RuntimeError(f"experiment Python does not exist: {python}")
    atomic_json(bundle / "environment.json", capture_environment(repo, python.absolute()))

    evidence = list(args.evidence)
    evidence.extend(manifests)
    for value, manifest_path in zip(values, manifests):
        extension = (((value.get("identity") or {}).get("design") or {}).get("extension") or {})
        base_output = extension.get("base_output")
        if base_output:
            base_manifest = Path(f"{base_output}.manifest.json")
            if base_manifest.is_file():
                evidence.append(base_manifest)
    if args.design_lock:
        lock_path = args.design_lock if args.design_lock.is_absolute() else repo / args.design_lock
        evidence.append(lock_path)
    evidence_records = copy_evidence(evidence, bundle / "evidence")

    data_records: dict[str, dict] = {}
    for value in values:
        for kind in ("personas", "prompts"):
            data = (value.get("identity") or {}).get("data") or {}
            path = Path(str(data[kind])).resolve(strict=True)
            key = str(path)
            if key in data_records:
                continue
            actual = sha256_file(path)
            expected = data.get(f"{kind}_sha256")
            if actual != expected:
                raise RuntimeError(f"data hash mismatch for {path}: expected {expected}, got {actual}")
            data_records[key] = {
                "path": key,
                "size": path.stat().st_size,
                "sha256": actual,
                "matches_manifest": True,
            }

    model_records = [
        artifact_record((repo / path) if not path.is_absolute() else path, hash_content=args.hash_large_artifacts)
        for path in args.model_artifact
    ]
    tokenizer_records = [
        artifact_record((repo / path) if not path.is_absolute() else path, hash_content=True)
        for path in args.tokenizer_file
    ]
    atomic_json(bundle / "artifacts.json", {
        "data": list(data_records.values()),
        "models": model_records,
        "tokenizer_files": tokenizer_records,
    })

    run_records = []
    for path, value in zip(manifests, values):
        extension = (((value.get("identity") or {}).get("design") or {}).get("extension") or {})
        run_records.append({
            "manifest": str(path),
            "run_id": value.get("run_id"),
            "probe": ((value.get("identity") or {}).get("instrument") or {}).get("probe"),
            "acquisition_digest": manifest_digest(value),
            "repo_digest": (value.get("provenance") or {}).get("repo_digest"),
            "coordinates_sha256": (
                ((value.get("identity") or {}).get("design") or {})
            ).get("coordinates_sha256"),
            "base_run_id": extension.get("base_run_id"),
            "base_output": extension.get("base_output"),
            "expected_extension_cells": extension.get("extension_cells"),
        })
    bundle_record = {
        "schema_version": 1,
        "kind": "syco-preservation-bundle",
        "name": args.name,
        "captured_at": dt.datetime.now(dt.UTC).isoformat(),
        "repo": str(repo),
        "source_digests": source_digests,
        "source": source,
        "runs": run_records,
        "jobs": args.job,
        "evidence": evidence_records,
        "notes": [
            "Log files are point-in-time snapshots of live append-only jobs.",
            "Result JSONL files are intentionally not copied while acquisition is active.",
            "Re-run capture after completion to preserve final result checksums.",
        ],
    }
    atomic_json(bundle / "bundle.json", bundle_record)
    write_checksums(bundle)
    return bundle


def verify(snapshot: Path) -> int:
    """Verify every recorded file checksum in a study snapshot."""
    snapshot = snapshot.resolve(strict=True)
    checksum_path = snapshot / "checksums.sha256"
    failures = []
    checked = 0
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        target = snapshot / relative
        actual = sha256_file(target) if target.is_file() else None
        checked += 1
        if actual != expected:
            failures.append({"path": relative, "expected": expected, "actual": actual})
    if failures:
        for failure in failures:
            print(
                f"FAIL {failure['path']}: expected {failure['expected']}, "
                f"got {failure['actual']}"
            )
        return 1
    print(f"OK: {checked} files verified in {snapshot}")
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create", parents=[create_parser()], add_help=False)
    create.description = "Capture source, environment, artifacts, and run evidence"
    check = commands.add_parser("verify", help="verify a captured snapshot")
    check.add_argument("snapshot", type=Path)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "verify":
        return verify(args.snapshot)
    print(capture(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
