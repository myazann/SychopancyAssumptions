"""Artifact identities, atomic writes, and design reconstruction."""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from syco import grid, paths
from syco.data import load_personas, load_prompts


@dataclass(frozen=True)
class ArtifactPaths:
    root: Path
    dataset_digest: str
    labels_digest: str
    activations_digest: str
    probes_digest: str
    steering_digest: str
    evaluation_digest: str

    @property
    def dataset_dir(self) -> Path:
        return self.root / "dataset" / self.dataset_digest

    @property
    def labels_dir(self) -> Path:
        return self.root / "labels" / self.labels_digest

    @property
    def raw_labels(self) -> Path:
        return self.labels_dir / "raw.jsonl"

    @property
    def dataset(self) -> Path:
        return self.dataset_dir / "cells.parquet"

    @property
    def dataset_personas(self) -> Path:
        return self.dataset_dir / "personas.parquet"

    @property
    def dataset_prompts(self) -> Path:
        return self.dataset_dir / "prompts.parquet"

    @property
    def dataset_demographics(self) -> Path:
        return self.dataset_dir / "demographics.parquet"

    @property
    def dataset_manifest(self) -> Path:
        return self.dataset_dir / "manifest.json"

    @property
    def labels(self) -> Path:
        return self.labels_dir / "labels.parquet"

    @property
    def label_quality(self) -> Path:
        return self.labels_dir / "quality.json"

    @property
    def activations(self) -> Path:
        return self.root / "activations" / self.activations_digest

    @property
    def activation_rows(self) -> Path:
        return self.activations / "rows.parquet"

    @property
    def probes(self) -> Path:
        return self.root / "probes" / self.probes_digest

    @property
    def steering(self) -> Path:
        return self.root / "steering" / self.steering_digest / "scores.jsonl"

    @property
    def evaluation(self) -> Path:
        return self.root / "evaluation" / self.evaluation_digest


def paths_for(config, *, dry_run: bool = False) -> ArtifactPaths:
    root = config.root / "dry_run" if dry_run else config.root
    return ArtifactPaths(
        root=root,
        dataset_digest=config.stage_digest("dataset"),
        labels_digest=config.stage_digest("labels"),
        activations_digest=config.stage_digest("activations"),
        probes_digest=config.stage_digest("probes"),
        steering_digest=config.stage_digest("steering"),
        evaluation_digest=config.stage_digest("evaluation"),
    )


def resolve_input(value: str | None, default: Path) -> Path:
    if value is None:
        return default
    path = Path(value).expanduser()
    return path if path.is_absolute() else paths.ROOT / path


def build_design(config):
    """Rebuild the exact paired grid selected by the pipeline config."""
    d = config.design
    persona_path = resolve_input(d.personas_path, paths.PERSONA_PATH)
    prompt_path = resolve_input(d.prompts_path, paths.PROMPT_PATH)
    personas, diagnostics = load_personas(persona_path)
    prompts = load_prompts(prompt_path)
    common = {
        "persona_types": list(d.persona_types) if d.persona_types else None,
        "prompt_types": list(d.prompt_types),
        "include_no_persona": d.include_control,
        "n_reps": d.n_reps,
        "seed": d.seed,
    }
    if d.pairing == "fully_crossed":
        cells = grid.build_cells(
            personas, prompts, n_persona_ids=d.n_persona_ids,
            n_prompt_ids=d.n_prompt_ids, **common,
        )
    else:
        eligible_people, eligible_prompts = grid.eligible_design_ids(
            personas, prompts, persona_types=common["persona_types"],
            prompt_types=common["prompt_types"],
        )
        people = grid.stable_sample(
            eligible_people, d.n_persona_ids, d.seed, "probe-persona"
        )
        dilemmas = grid.stable_sample(
            eligible_prompts, d.n_prompt_ids, d.seed, "probe-prompt"
        )
        if (config.training.split.group_by == "two_axis"
                and not d.include_cross_axis):
            persona_partitions = stable_group_partitions(
                people, config.training.split, "persona_id"
            )
            prompt_partitions = stable_group_partitions(
                dilemmas, config.training.split, "prompt_id"
            )
            pairs = set()
            for partition in ("train", "validation", "test"):
                partition_people = [
                    value for value in people
                    if persona_partitions[value] == partition
                ]
                partition_prompts = [
                    value for value in dilemmas
                    if prompt_partitions[value] == partition
                ]
                pairs.update(balanced_sparse_pairs(
                    partition_people,
                    partition_prompts,
                    int(d.dilemmas_per_persona),
                    d.seed,
                ))
        else:
            pairs = balanced_sparse_pairs(
                people, dilemmas, int(d.dilemmas_per_persona), d.seed
            )
        cells = grid.build_cells(
            personas, prompts, persona_ids=people, prompt_ids=dilemmas,
            restrict_pairs=pairs, **common,
        )
    return cells, diagnostics, persona_path, prompt_path


def stable_group_partitions(values, split, tag: str) -> dict[str, str]:
    """Deterministically assign whole groups to train/validation/test."""
    ordered = sorted({str(value) for value in values}, key=lambda value:
                     hashlib.sha256(
                         f"{split.seed}|{tag}|{value}".encode()
                     ).hexdigest())
    n = len(ordered)
    if n < 3:
        raise ValueError(
            f"{tag} has only {n} group(s); leakage-safe train/validation/test "
            "needs at least three"
        )
    n_validation = max(1, round(n * split.validation))
    n_test = max(1, round(n * split.test))
    n_train = n - n_validation - n_test
    if n_train < 1:
        raise ValueError(f"not enough {tag} groups for the configured split")
    out = {}
    for value in ordered[:n_train]:
        out[value] = "train"
    for value in ordered[n_train:n_train + n_validation]:
        out[value] = "validation"
    for value in ordered[n_train + n_validation:]:
        out[value] = "test"
    return out


def balanced_sparse_pairs(
    persona_ids: list[str], prompt_ids: list[str], per_persona: int, seed: int
) -> set[tuple[str, str]]:
    """Deterministic near-regular bipartite sample.

    Every person receives exactly ``per_persona`` dilemmas and prompt degrees
    differ by at most one. Hash-shuffled axes prevent source-table order from
    determining the pair pattern.
    """
    if not persona_ids or not prompt_ids:
        return set()
    if not 0 < per_persona <= len(prompt_ids):
        raise ValueError("per_persona must be in [1, number of prompts]")

    def order(values, tag):
        return sorted(values, key=lambda value: hashlib.sha256(
            f"{seed}|{tag}|{value}".encode()).hexdigest())

    people = order(persona_ids, "pair-persona")
    prompts = order(prompt_ids, "pair-prompt")
    result = set()
    # Consecutive slices in one cyclic prompt order make the degrees exactly
    # balanced while still giving each person a distinct set.
    for index, persona_id in enumerate(people):
        start = (index * per_persona) % len(prompts)
        for offset in range(per_persona):
            result.add((persona_id, prompts[(start + offset) % len(prompts)]))
    return result


def coordinate(cell) -> dict:
    return {
        "persona_type": cell.persona.persona_type,
        "persona_id": cell.persona.persona_id,
        "prompt_type": cell.prompt.prompt_type,
        "prompt_id": cell.prompt.prompt_id,
        "rep": int(cell.rep),
    }


def _stable_id(parts) -> str:
    blob = json.dumps(list(parts), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


def cell_id(cell) -> str:
    """Identity of one framing-specific activation/label example."""
    return _stable_id(cell.key_parts)


def design_unit_id(cell) -> str:
    """Identity shared by original/flipped framings for steering sampling."""
    return _stable_id((cell.persona.persona_type, cell.persona.persona_id,
                       cell.prompt.prompt_id, cell.rep))


def label_key(config_digest: str, teacher_id: str, cell, instrument: str,
              replicate: int) -> str:
    """Identity of one teacher-specific labeling task."""
    return "|".join((
        config_digest, teacher_id, cell_id(cell), instrument, str(replicate)
    ))


def prompt_digest(messages, system: str = "") -> str:
    blob = system + "\x00" + "\x00".join(
        f"{m['role']}:{m['content']}" for m in messages
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:20]


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
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
    return path


def append_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _quarantine_truncated_tail(path)
    with path.open("a", encoding="utf-8") as handle:
        payload = "".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in rows
        )
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _quarantine_truncated_tail(path: Path) -> None:
    """Preserve and remove an unterminated crash fragment before appending."""
    if not path.is_file() or path.stat().st_size == 0:
        return
    with path.open("r+b") as handle:
        handle.seek(-1, os.SEEK_END)
        if handle.read(1) == b"\n":
            return
        end = handle.tell()
        cursor = end
        newline = -1
        while cursor > 0 and newline < 0:
            start = max(0, cursor - 64 * 1024)
            handle.seek(start)
            chunk = handle.read(cursor - start)
            offset = chunk.rfind(b"\n")
            if offset >= 0:
                newline = start + offset
            cursor = start
        tail_start = newline + 1
        handle.seek(tail_start)
        fragment = handle.read(end - tail_start)
        try:
            complete_row = json.loads(fragment)
        except (UnicodeDecodeError, json.JSONDecodeError):
            complete_row = None
        if isinstance(complete_row, dict):
            handle.seek(0, os.SEEK_END)
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
            return
        quarantine = path.with_name(path.name + ".truncated")
        with quarantine.open("ab") as rejected:
            rejected.write(fragment + b"\n")
            rejected.flush()
            os.fsync(rejected.fileno())
        handle.truncate(tail_start)
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    with path.open("rb") as handle:
        line_number = 0
        line = handle.readline()
        while line:
            line_number += 1
            next_line = handle.readline()
            try:
                row = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                if not next_line and not line.endswith(b"\n"):
                    break
                raise ValueError(
                    f"malformed JSONL at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(  # noqa: TRY004
                    f"JSONL row at {path}:{line_number} is not an object"
                )
            rows.append(row)
            line = next_line
    return rows


def canonical_attempts(rows: list[dict], key: str) -> tuple[list[dict], dict]:
    """Latest successful attempt wins; failed rows remain retryable."""
    selected, counts = {}, {}
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        counts[value] = counts.get(value, 0) + 1
        old = selected.get(value)
        if old is None or old.get("error") or not row.get("error"):
            selected[value] = row
    return list(selected.values()), {
        "attempts": len(rows),
        "units": len(selected),
        "retried": sum(v > 1 for v in counts.values()),
        "extra_attempts": sum(max(v - 1, 0) for v in counts.values()),
    }


def stage_manifest(config, stage: str, *, inputs=None, details=None) -> dict:
    return {
        "schema_version": 1,
        "stage": stage,
        "created_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "config_name": config.name,
        "config_digest": config.stage_digest(stage),
        "pipeline_digest": config.digest,
        "config": dataclasses.asdict(config),
        "inputs": inputs or {},
        "details": details or {},
    }


def require_manifest(path: Path, config, expected_stage: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"missing {expected_stage} manifest: {path}")
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("stage") != expected_stage:
        raise ValueError(f"{path} is not a {expected_stage} manifest")
    expected_digest = config.stage_digest(expected_stage)
    if manifest.get("config_digest") != expected_digest:
        raise ValueError(
            f"{expected_stage} artifact was built with config "
            f"{manifest.get('config_digest')}, current stage config is "
            f"{expected_digest}"
        )
    return manifest
