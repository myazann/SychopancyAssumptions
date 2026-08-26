"""Profile-driven orchestration behind ``python -m syco``."""
from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from syco.experiments import ExperimentProfile
from syco.manifest import build_manifest, load_manifest, write_manifest
from syco.model_registry import LLAMACPP_BACKEND, load_registry
from syco.store import canonical_rows, read_rows


@dataclass(frozen=True)
class GPU:
    index: int
    total_mib: int
    free_mib: int


@dataclass
class Running:
    spec: object
    gpu: GPU
    process: subprocess.Popen
    log_handle: object


def _command(profile: ExperimentProfile, spec, *extra: str) -> list[str]:
    return [sys.executable, "-m", "syco", "run", *profile.run_args(spec), *extra]


def _expected_manifest(profile: ExperimentProfile, registry, spec) -> dict:
    """Manifest a fresh run under the current profile would produce."""
    from scripts.run_assumptions import configured_spec, parse_args
    from syco.prompts import ProbeSpec

    args = parse_args(profile.run_args(spec))
    effective_spec = configured_spec(args, registry)
    probe = ProbeSpec(
        kind=args.probe,
        history_mode=args.history_mode,
        n_models=args.n_models,
    )
    return build_manifest(args=args, spec=effective_spec, probe=probe)


def query_gpus() -> list[GPU]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as err:
        raise RuntimeError("nvidia-smi is not installed; no NVIDIA GPUs are available") from err
    except subprocess.CalledProcessError as err:
        raise RuntimeError(f"nvidia-smi failed: {err.stderr.strip()}") from err
    gpus = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        try:
            index, total, free = (int(part.strip()) for part in line.split(","))
        except ValueError as err:
            raise RuntimeError(f"Unexpected nvidia-smi output: {line!r}") from err
        gpus.append(GPU(index, total, free))
    if not gpus:
        raise RuntimeError("nvidia-smi reported no GPUs")
    return gpus


@contextlib.contextmanager
def profile_lock(profile: ExperimentProfile):
    """Prevent two schedulers from targeting the same profile outputs."""
    import fcntl

    profile.results_dir.mkdir(parents=True, exist_ok=True)
    path = profile.results_dir / f".{profile.name}.run-all.lock"
    handle = path.open("a+")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as err:
            raise RuntimeError(
                f"another run-all process holds {path}; use `python -m syco status` "
                "instead of starting a concurrent writer"
            ) from err
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def plan(profile: ExperimentProfile) -> int:
    registry = load_registry()
    rc = 0
    for spec in profile.select_models(registry):
        print(f"\n=== {spec.alias}", flush=True)
        result = subprocess.run(_command(profile, spec, "--plan-only"))
        rc = rc or result.returncode
    return rc


def run_all(
    profile: ExperimentProfile,
    *,
    limit_per_model: int | None = None,
    wait_timeout_seconds: int | None = None,
) -> int:
    registry = load_registry()
    specs = profile.select_models(registry)
    non_gpu = [spec for spec in specs if spec.backend != LLAMACPP_BACKEND]
    if non_gpu:
        names = ", ".join(spec.alias for spec in non_gpu)
        raise RuntimeError(
            "run --all currently schedules enabled llama.cpp models only; "
            f"run these separately: {names}"
        )
    missing = [spec.alias for spec in specs if not spec.estimated_vram_mib]
    if missing:
        raise RuntimeError(
            "Missing resources.estimated_vram_mib in models.yaml for: "
            + ", ".join(missing)
        )

    initial_gpus = query_gpus()
    largest_gpu = max(gpu.total_mib for gpu in initial_gpus)
    impossible = [
        spec.alias for spec in specs
        if spec.estimated_vram_mib > largest_gpu
    ]
    if impossible:
        raise RuntimeError(
            f"No GPU can fit the configured VRAM requirement for: {', '.join(impossible)}"
        )

    queue = sorted(specs, key=lambda spec: spec.estimated_vram_mib, reverse=True)
    running: dict[int, Running] = {}
    results: dict[str, int] = {}
    poll = int(profile.execution.get("poll_seconds", 2))
    report_every = int(profile.execution.get("wait_report_seconds", 60))
    timeout = wait_timeout_seconds
    if timeout is None:
        timeout = profile.execution.get("wait_timeout_seconds")
    blocked_since = None
    last_report = 0.0
    stop_requested = False

    def stop_handler(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    old_handlers = {
        sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)
    }
    for sig in old_handlers:
        signal.signal(sig, stop_handler)

    profile.logs_dir.mkdir(parents=True, exist_ok=True)
    extra = [] if limit_per_model is None else ["--limit", str(limit_per_model)]
    print(f"profile: {profile.name} ({profile.path})")
    print("queue:   " + " ".join(spec.alias for spec in queue))

    try:
        with profile_lock(profile):
            while queue or running:
                if stop_requested:
                    print("interrupt received; terminating child model runs", flush=True)
                    for item in running.values():
                        item.process.terminate()
                    for item in running.values():
                        try:
                            item.process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            item.process.kill()
                            item.process.wait()
                        item.log_handle.close()
                    return 130

                for gpu_index, item in list(running.items()):
                    rc = item.process.poll()
                    if rc is None:
                        continue
                    item.log_handle.close()
                    results[item.spec.alias] = rc
                    print(
                        f"[{time.strftime('%H:%M:%S')}] finished "
                        f"{item.spec.alias:<20} GPU{gpu_index} exit={rc}",
                        flush=True,
                    )
                    del running[gpu_index]

                if not queue and not running:
                    break

                current = {gpu.index: gpu for gpu in query_gpus()}
                launched = False
                for gpu in current.values():
                    if gpu.index in running:
                        continue
                    fitting = next(
                        (spec for spec in queue
                         if spec.estimated_vram_mib <= gpu.free_mib),
                        None,
                    )
                    if fitting is None:
                        continue
                    queue.remove(fitting)
                    log_path = profile.log_for(fitting)
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    log_handle = log_path.open("a", encoding="utf-8")
                    env = dict(os.environ)
                    env["CUDA_VISIBLE_DEVICES"] = str(gpu.index)
                    command = _command(profile, fitting, *extra)
                    print(
                        f"[{time.strftime('%H:%M:%S')}] launching "
                        f"{fitting.alias:<20} GPU{gpu.index} "
                        f"free={gpu.free_mib}MiB need={fitting.estimated_vram_mib}MiB",
                        flush=True,
                    )
                    process = subprocess.Popen(
                        command,
                        env=env,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                    )
                    running[gpu.index] = Running(fitting, gpu, process, log_handle)
                    launched = True

                if launched or running:
                    blocked_since = None
                elif queue:
                    blocked_since = blocked_since or time.monotonic()
                    waited = time.monotonic() - blocked_since
                    now = time.monotonic()
                    if now - last_report >= report_every:
                        needed = min(spec.estimated_vram_mib for spec in queue)
                        free = max(gpu.free_mib for gpu in current.values())
                        print(
                            f"waiting for GPU memory: best free={free}MiB, "
                            f"smallest queued requirement={needed}MiB",
                            flush=True,
                        )
                        last_report = now
                    if timeout is not None and waited >= timeout:
                        raise RuntimeError(
                            f"timed out after {int(waited)}s waiting for GPU memory"
                        )
                time.sleep(poll)
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)

    print("\n=== summary")
    failed = False
    for spec in specs:
        rc = results.get(spec.alias, 1)
        failed |= rc != 0
        print(
            f"  {spec.alias:<20} exit={rc:<3} "
            f"output={profile.output_for(spec)} log={profile.log_for(spec)}"
        )
    return 1 if failed else 0


def status(profile: ExperimentProfile) -> int:
    registry = load_registry()
    specs = profile.select_models(registry)
    expected = len(profile.build_cells()[0])
    print(f"profile: {profile.name} | expected cells/model: {expected}")
    incomplete = False
    for spec in specs:
        path = profile.output_for(spec)
        manifest = load_manifest(path)
        expected_manifest = _expected_manifest(profile, registry, spec)
        manifest_state = "current"
        if manifest is None:
            manifest_state = "missing-manifest"
        elif manifest.get("run_id") != expected_manifest.get("run_id"):
            manifest_state = "stale-manifest"
        attempts = read_rows(path)
        rows, diagnostics = canonical_rows(attempts)
        successes = sum(not row.get("error") for row in rows)
        errors = sum(bool(row.get("error")) for row in rows)
        missing = max(0, expected - successes)
        incomplete |= bool(missing or errors or manifest_state != "current")
        print(
            f"  {spec.alias:<20} success={successes:<6} missing={missing:<6} "
            f"errors={errors:<4} attempts={diagnostics['attempts']:<6} "
            f"manifest={manifest_state:<16} {path}"
        )
    return 1 if incomplete else 0


def _comparable_identity(manifest: dict) -> dict:
    identity = dict(manifest.get("identity") or {})
    identity.pop("model", None)
    return identity


def merge(profile: ExperimentProfile, *, allow_partial: bool = False) -> int:
    registry = load_registry()
    specs = profile.select_models(registry)
    expected = len(profile.build_cells()[0])
    merged_rows = []
    reference_identity = None
    input_manifests = []

    for spec in specs:
        path = profile.output_for(spec)
        if not path.is_file():
            raise RuntimeError(f"missing model output: {path}")
        manifest = load_manifest(path)
        if manifest is None:
            raise RuntimeError(f"missing run manifest: {path}.manifest.json")
        expected_manifest = _expected_manifest(profile, registry, spec)
        if manifest.get("run_id") != expected_manifest.get("run_id"):
            raise RuntimeError(
                f"{spec.alias} output belongs to run {manifest.get('run_id')}, "
                f"but the current profile/code is run {expected_manifest.get('run_id')}"
            )
        comparable = _comparable_identity(manifest)
        if reference_identity is None:
            reference_identity = comparable
        elif comparable != reference_identity:
            raise RuntimeError(
                f"experiment manifest mismatch for {spec.alias}; refusing to merge"
            )
        attempts = read_rows(path)
        rows, diagnostics = canonical_rows(attempts)
        wrong_run = [row for row in rows if row.get("run_id") != manifest["run_id"]]
        if wrong_run:
            raise RuntimeError(f"{path} contains rows from another run")
        errors = [row for row in rows if row.get("error")]
        successes = [row for row in rows if not row.get("error")]
        if errors and not allow_partial:
            raise RuntimeError(f"{spec.alias} has {len(errors)} unresolved error cell(s)")
        if len(successes) != expected and not allow_partial:
            raise RuntimeError(
                f"{spec.alias} is incomplete: {len(successes)}/{expected} successful cells"
            )
        print(
            f"  {spec.alias:<20} cells={len(successes):<6} "
            f"superseded_attempts={diagnostics['extra_attempts']}"
        )
        merged_rows.extend(successes)
        input_manifests.append({"model": spec.alias, "run_id": manifest["run_id"]})

    target = profile.merged_output()
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in merged_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    write_manifest(target, {
        "schema_version": 1,
        "kind": "merged",
        "profile": profile.name,
        "partial": bool(allow_partial),
        "inputs": input_manifests,
        "cells": len(merged_rows),
    })
    print(f"merged {len(merged_rows)} canonical row(s) -> {target}")
    return 0


def parse_all(profile: ExperimentProfile, extra: list[str] | None = None) -> int:
    from scripts.parse_assumptions import main as parse_main

    registry = load_registry()
    rc = 0
    for spec in profile.select_models(registry):
        path = profile.output_for(spec)
        if not path.is_file():
            print(f"missing: {path}", file=sys.stderr)
            rc = 1
            continue
        print(f"\n=== {spec.alias}")
        rc = parse_main([str(path), *(extra or [])]) or rc
    return rc


def summarize_all(profile: ExperimentProfile, extra: list[str] | None = None) -> int:
    from scripts.summarize_assumptions import main as summarize_main

    registry = load_registry()
    rc = 0
    for spec in profile.select_models(registry):
        raw = profile.output_for(spec)
        parsed = Path(str(raw).removesuffix(".jsonl") + "_assumptions.parquet")
        if not parsed.is_file():
            print(f"missing: {parsed}", file=sys.stderr)
            rc = 1
            continue
        print(f"\n=== {spec.alias}")
        rc = summarize_main([str(parsed), *(extra or [])]) or rc
    return rc


def topics_all(profile: ExperimentProfile, extra: list[str] | None = None) -> int:
    from scripts.topic_assumptions import main as topics_main

    registry = load_registry()
    rc = 0
    for spec in profile.select_models(registry):
        raw = profile.output_for(spec)
        parsed = Path(str(raw).removesuffix(".jsonl") + "_assumptions.parquet")
        if not parsed.is_file():
            print(f"missing: {parsed}", file=sys.stderr)
            rc = 1
            continue
        print(f"\n=== {spec.alias}")
        rc = topics_main([str(parsed), *(extra or [])]) or rc
    return rc


def smoke(profile: ExperimentProfile, model: str | None = None) -> int:
    from scripts.parse_assumptions import main as parse_main
    from scripts.run_assumptions import main as run_main
    from scripts.summarize_assumptions import main as summarize_main
    from scripts.topic_assumptions import main as topics_main

    registry = load_registry()
    spec = registry.get(model) if model else profile.select_models(registry)[0]
    smoke_dir = profile.results_dir / "smoke"
    output = smoke_dir / f"{spec.safe_dir_name()}_{profile.name}.jsonl"
    args = [
        *profile.run_args(spec),
        "--out", str(output),
        "--n-personas", "1",
        "--n-prompts", "1",
        "--dry-run",
        "--overwrite",
    ]
    rc = run_main(args)
    if rc:
        return rc
    rc = parse_main([str(output), "--cells"])
    if rc:
        return rc
    parsed = Path(str(output).removesuffix(".jsonl") + "_assumptions.parquet")
    rc = summarize_main([str(parsed)])
    if rc:
        return rc
    # The mock backend draws its labels from a fixed list, so nothing the
    # topic model finds here means anything. What is being checked is that the
    # whole path runs offline: n-grams always, and BERTopic too when the
    # optional stack is installed -- and that it says why when it is not.
    return topics_main([str(parsed), "--top", "5", "--no-write"])


def doctor(profile: ExperimentProfile) -> int:
    registry = load_registry()
    specs = profile.select_models(registry)
    checks = {
        "yaml": "core configuration",
        "pandas": "data loading",
        "pyarrow": "default parser output",
        "huggingface_hub": "GGUF resolution",
        "transformers": "chat templates",
    }
    if any(spec.backend == LLAMACPP_BACKEND for spec in specs):
        checks["llama_cpp"] = "enabled GGUF models"
    if any(spec.backend == "openai" for spec in specs):
        checks["openai"] = "enabled OpenAI models"
    if any(spec.backend == "anthropic" for spec in specs):
        checks["anthropic"] = "enabled Anthropic models"

    failed = False
    print(f"profile: {profile.name} ({profile.path})")
    for module, purpose in checks.items():
        ok = importlib.util.find_spec(module) is not None
        failed |= not ok
        print(f"  {'ok' if ok else 'MISSING':<7} Python module {module:<18} {purpose}")
    from syco.topics import topics_available

    available, why = topics_available()
    print(f"  {'ok' if available else 'note':<7} topic model        "
          f"{'bertopic + sentence-transformers' if available else why}")
    try:
        cells, diagnostics = profile.build_cells()
        unusable = int((~diagnostics.usable).sum()) if len(diagnostics) else 0
        print(f"  ok      data grid: {len(cells)} cells/model, unusable personas={unusable}")
    except Exception as err:
        failed = True
        print(f"  FAILED  data/profile: {type(err).__name__}: {err}")
    try:
        gpus = query_gpus()
        summary = ", ".join(
            f"GPU{gpu.index} {gpu.free_mib}/{gpu.total_mib}MiB free" for gpu in gpus
        )
        print(f"  ok      {summary}")
    except RuntimeError as err:
        if any(spec.backend == LLAMACPP_BACKEND for spec in specs):
            failed = True
        print(f"  {'FAILED' if failed else 'note':<7} {err}")
    for spec in specs:
        if spec.backend == LLAMACPP_BACKEND and not spec.estimated_vram_mib:
            failed = True
            print(f"  FAILED  {spec.alias}: no estimated_vram_mib")
    return 1 if failed else 0
