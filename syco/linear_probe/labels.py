"""Multi-teacher GGUF labeling with strict schemas and safe retry history."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import random
from collections import Counter, deque
from dataclasses import replace
from pathlib import Path

import pandas as pd

from syco.linear_probe.artifacts import (
    append_jsonl,
    atomic_json,
    cell_id,
    label_key,
    read_jsonl,
    sha256_file,
    stage_manifest,
)
from syco.linear_probe.dataset import (
    _atomic_parquet,
    freeze_dataset,
    load_frozen_cells,
)
from syco.linear_probe.prompts import build_label_prompt
from syco.model_registry import load_registry
from syco.models import Conversation, build_adapter
from syco.prompts import STRUCTURED_CONTAINER, STRUCTURED_DIMENSIONS


class InvalidLabel(ValueError):
    """The completion is not exactly the registered label schema."""


def _tokenizer_snapshot_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    for source in sorted(path.rglob("*")):
        if not source.is_file():
            continue
        digest.update(str(source.relative_to(path)).encode())
        digest.update(b"\0")
        with source.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _resolve_pinned_label_spec(spec, teacher):
    """Resolve configured GGUF/tokenizer commits to immutable local paths."""
    if spec.quantization.format != "gguf":
        if any((teacher.model_file, teacher.model_revision,
                teacher.model_sha256, teacher.tokenizer_revision)):
            raise ValueError(
                "labeling model_file/revision/SHA pinning is currently implemented "
                "for the GGUF label backend"
            )
        return spec, spec.provenance()

    from huggingface_hub import hf_hub_download, snapshot_download

    from syco.model_registry import split_hf_gguf_ref

    repo_id, pinned_file = split_hf_gguf_ref(spec.ref)
    filename = teacher.model_file or pinned_file or spec.quantization.resolved_file
    if not filename:
        raise RuntimeError("the GGUF label model has no resolved filename")
    weight_path = Path(hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=teacher.model_revision,
    ))
    resolved_weight = weight_path.resolve()
    # Hugging Face stores LFS objects under their content SHA-256. Fall back to
    # hashing for a nonstandard/local cache layout.
    blob_name = resolved_weight.name.lower()
    actual_sha = (
        blob_name
        if len(blob_name) == 64
        and all(character in "0123456789abcdef" for character in blob_name)
        else sha256_file(resolved_weight)
    )
    if (teacher.model_sha256 is not None
            and actual_sha != teacher.model_sha256.lower()):
        raise RuntimeError(
            f"label weight SHA mismatch: expected {teacher.model_sha256}, "
            f"resolved {actual_sha}"
        )

    tokenizer_id = spec.tokenizer_id or spec.hf_id
    if not tokenizer_id:
        raise RuntimeError("the label model has no tokenizer repository")
    tokenizer_path = Path(snapshot_download(
        repo_id=tokenizer_id,
        revision=teacher.tokenizer_revision,
        allow_patterns=[
            "*.json", "*.jinja", "*.txt", "*.model", "*.tiktoken",
        ],
    ))
    provenance = spec.provenance()
    provenance.update({
        "model_repository": repo_id,
        "model_file": filename,
        "label_teacher_id": teacher.id,
        "model_revision_requested": teacher.model_revision,
        "model_revision_resolved": weight_path.parent.name,
        "model_weight_sha256": actual_sha,
        "model_weight_bytes": weight_path.stat().st_size,
        "tokenizer_repository": tokenizer_id,
        "tokenizer_revision_requested": teacher.tokenizer_revision,
        "tokenizer_revision_resolved": tokenizer_path.name,
        "tokenizer_snapshot_sha256": _tokenizer_snapshot_fingerprint(
            tokenizer_path
        ),
    })
    local_spec = replace(
        spec,
        ref=str(weight_path),
        tokenizer_id=str(tokenizer_path),
        quantization=replace(spec.quantization, resolved_file=filename),
    )
    return local_spec, provenance


def raw_label_paths(artifacts, teacher_id: str | None = None) -> list:
    paths = []
    if artifacts.raw_labels.is_file():
        paths.append(artifacts.raw_labels)
    paths.extend(sorted(artifacts.raw_labels.parent.glob("raw.shard-*-of-*.jsonl")))
    if teacher_id is None:
        paths.extend(sorted(
            artifacts.raw_labels.parent.glob("raw.teacher-*.jsonl")
        ))
    else:
        single = artifacts.raw_labels.parent / f"raw.teacher-{teacher_id}.jsonl"
        if single.is_file():
            paths.append(single)
        paths.extend(sorted(artifacts.raw_labels.parent.glob(
            f"raw.teacher-{teacher_id}.shard-*-of-*.jsonl"
        )))
    return paths


def read_all_raw_labels(artifacts, teacher_id: str | None = None) -> list[dict]:
    rows = []
    for path in raw_label_paths(artifacts, teacher_id):
        rows.extend(read_jsonl(path))
    return rows


def shard_raw_path(artifacts, teacher_id: str, shard_index: int,
                   num_shards: int):
    if num_shards == 1:
        return artifacts.raw_labels.parent / f"raw.teacher-{teacher_id}.jsonl"
    return artifacts.raw_labels.parent / (
        f"raw.teacher-{teacher_id}.shard-{shard_index:03d}-of-"
        f"{num_shards:03d}.jsonl"
    )


def teacher_work_manifest_path(artifacts, teacher_id: str) -> Path:
    return artifacts.labels_dir / f"work_manifest.teacher-{teacher_id}.json"


def _select_teacher(labeling, teacher_id: str | None):
    if teacher_id is None:
        if len(labeling.models) != 1:
            choices = ", ".join(model.id for model in labeling.models)
            raise ValueError(
                "multiple label teachers are configured; choose one with "
                f"--teacher ({choices})"
            )
        return labeling.models[0]
    matches = [model for model in labeling.models if model.id == teacher_id]
    if not matches:
        choices = ", ".join(model.id for model in labeling.models)
        raise ValueError(f"unknown label teacher {teacher_id!r}; choose {choices}")
    return matches[0]


def strict_parse_label(raw: str, instrument: str) -> list[dict]:
    """Parse an exact JSON object; never repair, coerce, rescale, or clip."""
    if instrument not in STRUCTURED_DIMENSIONS:
        raise InvalidLabel(f"unknown instrument {instrument!r}")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise InvalidLabel(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        payload = json.loads(raw.strip(), object_pairs_hook=unique_object)
    except InvalidLabel:
        raise
    except (TypeError, json.JSONDecodeError) as exc:
        raise InvalidLabel(f"not a bare valid JSON object: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != {"mental_model"}:
        raise InvalidLabel("top level must contain exactly 'mental_model'")
    mental = payload["mental_model"]
    container = STRUCTURED_CONTAINER[instrument]
    if not isinstance(mental, dict) or set(mental) != {container}:
        raise InvalidLabel(f"mental_model must contain exactly {container!r}")
    beliefs = mental[container]
    expected = tuple(STRUCTURED_DIMENSIONS[instrument])
    if not isinstance(beliefs, dict) or set(beliefs) != set(expected):
        missing = sorted(set(expected) - set(beliefs) if isinstance(beliefs, dict)
                         else set(expected))
        extra = sorted(set(beliefs) - set(expected) if isinstance(beliefs, dict)
                       else set())
        raise InvalidLabel(f"dimension keys differ (missing={missing}, extra={extra})")

    rows = []
    for dimension in expected:
        entry = beliefs[dimension]
        if not isinstance(entry, dict) or set(entry) != {"score", "explanation"}:
            raise InvalidLabel(
                f"{dimension} must contain exactly score and explanation"
            )
        score = entry["score"]
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise InvalidLabel(f"{dimension}.score must be a JSON number")
        score = float(score)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise InvalidLabel(f"{dimension}.score is not finite in [0,1]")
        explanation = entry["explanation"]
        if not isinstance(explanation, str) or not explanation.strip():
            raise InvalidLabel(
                f"{dimension}.explanation must be a non-empty string"
            )
        rows.append({
            "dimension": dimension,
            "score": score,
            "explanation": explanation.strip(),
        })
    return rows


def _mock_completion(instrument: str, seed_text: str) -> str:
    rng = random.Random(int(hashlib.sha256(seed_text.encode()).hexdigest(), 16))
    container = STRUCTURED_CONTAINER[instrument]
    payload = {"mental_model": {container: {}}}
    for dimension in STRUCTURED_DIMENSIONS[instrument]:
        payload["mental_model"][container][dimension] = {
            "score": round(rng.random(), 4),
            "explanation": f"Synthetic dry-run rationale for {dimension}.",
        }
    return json.dumps(payload, ensure_ascii=False)


def _completed(rows: list[dict], labeling) -> tuple[set[str], Counter, set[str]]:
    done, attempts, latest = set(), Counter(), {}
    for row in rows:
        key = row.get("label_key")
        if not key:
            continue
        attempts[key] += 1
        latest[key] = row
        if not row.get("error") and row.get("strict_valid") is True:
            done.add(key)
    exhausted = set()
    for key, row in latest.items():
        if key in done:
            continue
        # A runtime/provider error may be transient. At temperature zero, an
        # otherwise successful but schema-invalid completion is deterministic
        # and resubmitting the identical prompt only wastes GPU time.
        limit = (
            labeling.max_attempts
            if row.get("error") or labeling.temperature > 0 else 1
        )
        if attempts[key] >= limit:
            exhausted.add(key)
    return done, attempts, exhausted


def run_labeling(config, artifacts, *, dry_run: bool = False,
                 limit: int | None = None, shard_index: int = 0,
                 num_shards: int = 1, teacher_id: str | None = None) -> dict:
    """Generate raw labels. A task is complete only after strict validation."""
    freeze_dataset(config, artifacts)
    cells, table = load_frozen_cells(config, artifacts)
    teacher = _select_teacher(config.labeling, teacher_id)
    if num_shards <= 0 or not 0 <= shard_index < num_shards:
        raise ValueError("label shard must satisfy 0 <= shard_index < num_shards")
    raw_path = shard_raw_path(
        artifacts, teacher.id, shard_index, num_shards
    )
    existing = read_all_raw_labels(artifacts, teacher.id)
    done, attempt_counts, exhausted = _completed(existing, config.labeling)
    labels_digest = config.stage_digest("labels")

    queue = deque()
    for cell, row in zip(cells, table.itertuples(index=False)):
        for instrument in config.labeling.instruments:
            for label_rep in range(config.labeling.replicates):
                key = label_key(
                    labels_digest, teacher.id, cell, instrument, label_rep
                )
                assigned = int(hashlib.sha256(key.encode()).hexdigest(), 16) % num_shards
                if assigned != shard_index:
                    continue
                if key in done or key in exhausted:
                    continue
                queue.append((cell, row, instrument, label_rep, key))
    if limit is not None:
        queue = deque(list(queue)[:limit])
    planned = len(queue)
    if not queue:
        return {
            "planned": 0,
            "written": 0,
            "valid": 0,
            "invalid": 0,
            "raw_path": str(raw_path),
            "teacher_id": teacher.id,
        }

    spec = None
    adapter = None
    plan = None
    model_provenance = {
        "backend": "mock",
        "model_id": teacher.model,
        "label_teacher_id": teacher.id,
    }
    if not dry_run:
        registry = load_registry()
        spec = registry.get(teacher.model)
        if spec.quantization.format == "gguf" and not teacher.model_file:
            spec = registry.with_resolved_quant(spec)
        elif spec.quantization.format == "gguf":
            spec = replace(
                spec,
                quantization=replace(
                spec.quantization,
                    resolved_file=teacher.model_file,
                ),
            )
        spec = replace(
            spec,
            temperature=config.labeling.temperature,
            top_p=config.labeling.top_p,
            max_output_tokens=config.labeling.max_output_tokens,
            batch_size=config.labeling.batch_size,
        )
        spec, model_provenance = _resolve_pinned_label_spec(
            spec, teacher
        )
    provenance_digest = hashlib.sha256(json.dumps(
        model_provenance, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    work_manifest_path = teacher_work_manifest_path(artifacts, teacher.id)
    work_details = {
        "dataset_manifest_sha256": sha256_file(artifacts.dataset_manifest),
        "label_teacher_id": teacher.id,
        "label_model_alias": teacher.model,
        "model_provenance": model_provenance,
        "model_provenance_digest": provenance_digest,
        "dry_run": dry_run,
    }
    if work_manifest_path.is_file():
        from syco.linear_probe.artifacts import require_manifest

        work_manifest = require_manifest(
            work_manifest_path, config, "labels_work"
        )
        if work_manifest.get("details") != work_details:
            raise RuntimeError(
                "existing raw labels were produced by different model weights, "
                "tokenizer files, or frozen data; use a separate output directory"
            )
    else:
        atomic_json(
            work_manifest_path,
            stage_manifest(config, "labels_work", details=work_details),
        )

    if not dry_run:
        adapter = build_adapter(spec)
        plan = adapter.thinking_plan(config.labeling.thinking)
        if not plan.standardized:
            adapter.close()
            raise RuntimeError(
                f"label model cannot enforce thinking={config.labeling.thinking}: "
                f"{plan.applied}"
            )

    written = valid = invalid = 0
    # GGUF uses a shared KV cache and cannot batch; HF/API adapters may.
    batch_size = (config.labeling.batch_size
                  if adapter is not None and adapter.batches else 1)
    try:
        while queue:
            batch = [queue.popleft() for _ in range(min(batch_size, len(queue)))]
            conversations, prompts = [], []
            for cell, dataset_row, instrument, _, _ in batch:
                text = build_label_prompt(
                    instrument, cell.persona.messages, cell.prompt.text
                )
                digest = hashlib.sha256(text.encode()).hexdigest()[:20]
                frozen_digest = getattr(
                    dataset_row, f"label_prompt_digest_{instrument}"
                )
                if digest != frozen_digest:
                    raise RuntimeError(
                        f"label prompt drift for {cell_id(cell)}/{instrument}: "
                        "the current builder no longer matches the frozen design"
                    )
                prompts.append(text)
                conversations.append(Conversation(
                    messages=({"role": "user", "content": text},), system=""
                ))
            try:
                outputs = (
                    [_mock_completion(item[2], item[4]) for item in batch]
                    if dry_run else adapter.chat_batch(conversations, plan=plan)
                )
                if len(outputs) != len(batch):
                    raise RuntimeError(
                        f"label adapter returned {len(outputs)} outputs for "
                        f"{len(batch)} inputs"
                    )
                errors = [""] * len(batch)
            except Exception as exc:  # noqa: BLE001 -- persist per-cell model failures
                outputs = [""] * len(batch)
                errors = [f"{type(exc).__name__}: {exc}"] * len(batch)

            records = []
            retry = []
            for item, text, raw, error in zip(batch, prompts, outputs, errors):
                cell, dataset_row, instrument, label_rep, key = item
                parse_error = ""
                if not error:
                    try:
                        strict_parse_label(raw, instrument)
                    except InvalidLabel as exc:
                        parse_error = str(exc)
                is_valid = not error and not parse_error
                attempt = attempt_counts[key] + 1
                attempt_counts[key] = attempt
                record = {
                    "schema_version": 1,
                    "config_digest": labels_digest,
                    "pipeline_digest": config.digest,
                    "label_key": key,
                    "cell_id": cell_id(cell),
                    "row_index": int(dataset_row.row_index),
                    "persona_type": cell.persona.persona_type,
                    "persona_id": cell.persona.persona_id,
                    "prompt_type": cell.prompt.prompt_type,
                    "prompt_id": cell.prompt.prompt_id,
                    "rep": cell.rep,
                    "split": dataset_row.split,
                    "instrument": instrument,
                    "label_rep": label_rep,
                    "attempt": attempt,
                    "label_teacher_id": teacher.id,
                    "label_model": teacher.model,
                    "label_model_provenance_digest": provenance_digest,
                    "temperature": config.labeling.temperature,
                    "thinking": config.labeling.thinking,
                    "thinking_applied": "mock" if dry_run else plan.applied,
                    "prompt_digest": hashlib.sha256(text.encode()).hexdigest()[:20],
                    "raw": raw,
                    "strict_valid": is_valid,
                    "validation_error": parse_error,
                    "error": error,
                    "timestamp": dt.datetime.now(dt.UTC).isoformat(
                        timespec="seconds"
                    ),
                }
                records.append(record)
                written += 1
                if is_valid:
                    valid += 1
                else:
                    invalid += 1
                    retry_limit = (
                        config.labeling.max_attempts
                        if error or config.labeling.temperature > 0 else 1
                    )
                    if attempt < retry_limit:
                        retry.append(item)
            append_jsonl(raw_path, records)
            queue.extend(retry)
    finally:
        if adapter is not None:
            adapter.close()

    raw_rows = read_all_raw_labels(artifacts, teacher.id)
    manifest = stage_manifest(
        config,
        "labels_raw",
        inputs={
            "dataset_manifest_sha256": sha256_file(artifacts.dataset_manifest),
        },
        details={
            "tasks_initially_pending": planned,
            "rows_written_this_run": written,
            "valid_this_run": valid,
            "invalid_this_run": invalid,
            "total_attempt_rows": len(raw_rows),
            "dry_run": dry_run,
            "shard_index": shard_index,
            "num_shards": num_shards,
            "label_teacher_id": teacher.id,
            "label_model_alias": teacher.model,
            "model": model_provenance,
            "model_provenance_digest": provenance_digest,
        },
    )
    atomic_json(raw_path.with_suffix(".manifest.json"), manifest)
    return {"planned": planned, "written": written, "valid": valid,
            "invalid": invalid, "raw_path": str(raw_path),
            "teacher_id": teacher.id}


def _teacher_agreement(labels: pd.DataFrame, teachers) -> dict:
    """Pairwise score agreement for auditing an ensemble before aggregation."""
    valid = labels[labels.valid_completion & labels.score.notna()]
    teacher_ids = [teacher.id for teacher in teachers]
    if len(teacher_ids) < 2 or valid.empty:
        return {}
    agreement = {}
    for dimension, group in valid.groupby("dimension"):
        # Replicates are first averaged within teacher; cross-teacher agreement
        # must not mistake repeated calls from one teacher for independent votes.
        pivot = group.pivot_table(
            index="cell_id", columns="label_teacher_id", values="score",
            aggfunc="mean",
        )
        pairs = {}
        for left_index, left in enumerate(teacher_ids):
            for right in teacher_ids[left_index + 1:]:
                if left not in pivot or right not in pivot:
                    continue
                complete = pivot[[left, right]].dropna()
                if complete.empty:
                    continue
                difference = (complete[left] - complete[right]).abs()
                correlation = (
                    complete[left].corr(complete[right])
                    if len(complete) >= 2 else float("nan")
                )
                pairs[f"{left}__{right}"] = {
                    "n": len(complete),
                    "mean_absolute_difference": float(difference.mean()),
                    "median_absolute_difference": float(difference.median()),
                    "fraction_absolute_difference_above_0_2": float(
                        (difference > .2).mean()
                    ),
                    "pearson": (
                        float(correlation) if math.isfinite(correlation) else None
                    ),
                }
        agreement[dimension] = pairs
    return agreement


def parse_labels(config, artifacts) -> tuple[pd.DataFrame, dict]:
    """Derive tidy strict labels while retaining quarantined completions."""
    from syco.linear_probe.artifacts import require_manifest

    work_manifests = {}
    provenance_by_teacher = {}
    for teacher in config.labeling.models:
        work_path = teacher_work_manifest_path(artifacts, teacher.id)
        work_manifest = require_manifest(work_path, config, "labels_work")
        details = work_manifest.get("details") or {}
        if details.get("label_teacher_id") != teacher.id:
            raise ValueError(
                f"label work manifest for {teacher.id!r} has the wrong teacher ID"
            )
        provenance_digest = details.get("model_provenance_digest")
        if not provenance_digest:
            raise ValueError(
                f"label work manifest for {teacher.id!r} has no provenance digest"
            )
        work_manifests[teacher.id] = work_path
        provenance_by_teacher[teacher.id] = provenance_digest
    raw_paths = raw_label_paths(artifacts)
    raw = read_all_raw_labels(artifacts)
    if not raw:
        raise FileNotFoundError(f"no raw labels at {artifacts.raw_labels}")
    labels_digest = config.stage_digest("labels")
    foreign = sorted({
        str(row.get("config_digest", "<missing>"))
        for row in raw
        if row.get("config_digest") != labels_digest
    })
    if foreign:
        raise ValueError(
            "raw label directory contains rows from another configuration "
            f"({foreign}); move them to a separate artifact root"
        )
    bad_provenance = sum(
        row.get("label_teacher_id") not in provenance_by_teacher
        or row.get("label_model_provenance_digest")
        != provenance_by_teacher.get(row.get("label_teacher_id"))
        for row in raw
    )
    if bad_provenance:
        raise ValueError(
            f"{bad_provenance} raw label row(s) do not match the pinned label "
            "model/tokenizer provenance"
        )

    cells, dataset = load_frozen_cells(config, artifacts)
    expected_tasks = {}
    for cell, dataset_row in zip(cells, dataset.itertuples(index=False)):
        for teacher in config.labeling.models:
            for instrument in config.labeling.instruments:
                for label_rep in range(config.labeling.replicates):
                    key = label_key(
                        labels_digest, teacher.id, cell, instrument, label_rep
                    )
                    expected_tasks[key] = {
                        "cell_id": cell_id(cell),
                        "row_index": int(dataset_row.row_index),
                        "persona_type": cell.persona.persona_type,
                        "persona_id": cell.persona.persona_id,
                        "prompt_type": cell.prompt.prompt_type,
                        "prompt_id": cell.prompt.prompt_id,
                        "rep": int(cell.rep),
                        "split": dataset_row.split,
                        "instrument": instrument,
                        "label_rep": label_rep,
                        "label_teacher_id": teacher.id,
                        "label_model": teacher.model,
                        "prompt_digest": getattr(
                            dataset_row, f"label_prompt_digest_{instrument}"
                        ),
                    }
    expected_keys = set(expected_tasks)
    unexpected_keys = sorted({
        str(row.get("label_key", "<missing>"))
        for row in raw
        if row.get("label_key") not in expected_keys
    })
    if unexpected_keys:
        preview = unexpected_keys[:5]
        raise ValueError(
            f"raw labels contain {len(unexpected_keys)} task key(s) outside the "
            f"frozen design; first entries: {preview}"
        )
    identity_fields = (
        "cell_id", "row_index", "persona_type", "persona_id", "prompt_type",
        "prompt_id", "rep", "split", "instrument", "label_rep",
        "label_teacher_id", "label_model", "prompt_digest",
    )
    for row_number, row in enumerate(raw, start=1):
        expected = expected_tasks[row["label_key"]]
        mismatched = [
            field for field in identity_fields
            if row.get(field) != expected[field]
        ]
        if mismatched:
            raise ValueError(
                f"raw label row {row_number} disagrees with the frozen task on "
                f"fields {mismatched}"
            )
    selected, attempts = {}, Counter()
    for row in raw:
        key = row.get("label_key")
        if not key:
            continue
        attempts[key] += 1
        previous = selected.get(key)
        if previous is None or (not previous.get("strict_valid")
                                or row.get("strict_valid")):
            # Once valid, an invalid later attempt cannot displace it. Among
            # attempts with the same validity, the latest one is authoritative.
            selected[key] = row
    canonical = list(selected.values())
    retry = {
        "attempts": len(raw),
        "units": len(canonical),
        "retried": sum(value > 1 for value in attempts.values()),
        "extra_attempts": sum(max(value - 1, 0) for value in attempts.values()),
    }
    records = []
    completion_counts = Counter()
    for row in canonical:
        task = expected_tasks[row["label_key"]]
        parsed = []
        parse_error = row.get("error") or row.get("validation_error") or ""
        if not parse_error:
            try:
                parsed = strict_parse_label(
                    row.get("raw", ""), task["instrument"]
                )
            except InvalidLabel as exc:
                parse_error = str(exc)
        valid = not parse_error and len(parsed) == len(
            STRUCTURED_DIMENSIONS[task["instrument"]]
        )
        completion_counts[(
            task["label_teacher_id"], task["instrument"], valid
        )] += 1
        by_dimension = {item["dimension"]: item for item in parsed}
        for dimension in STRUCTURED_DIMENSIONS[task["instrument"]]:
            item = by_dimension.get(dimension, {})
            records.append({
                "config_digest": labels_digest,
                "pipeline_digest": config.digest,
                "label_key": row["label_key"],
                **task,
                "dimension": dimension,
                "score": item.get("score"),
                "explanation": item.get("explanation", ""),
                "valid_completion": valid,
                "validation_error": parse_error,
                "attempt": row.get("attempt"),
            })
    labels = pd.DataFrame(records).sort_values(
        ["row_index", "instrument", "label_teacher_id", "label_rep", "dimension"]
    )
    artifacts.labels.parent.mkdir(parents=True, exist_ok=True)
    _atomic_parquet(labels, artifacts.labels)

    valid_labels = labels[labels.valid_completion & labels.score.notna()]
    summaries = {}
    for dimension, group in valid_labels.groupby("dimension"):
        summaries[dimension] = {
            "n": len(group),
            "mean": float(group.score.mean()),
            "std": float(group.score.std()),
            "below_0_3": int((group.score < .3).sum()),
            "above_0_7": int((group.score > .7).sum()),
        }
    expected_completions = len(expected_keys)
    observed_keys = set(selected)
    quality = {
        "config_digest": labels_digest,
        "pipeline_digest": config.digest,
        "attempt_history": retry,
        "canonical_completions": len(canonical),
        "valid_completions": sum(
            count for key, count in completion_counts.items() if key[-1]
        ),
        "invalid_completions": sum(
            count for key, count in completion_counts.items() if not key[-1]
        ),
        "expected_completions": expected_completions,
        "missing_completions": len(expected_keys - observed_keys),
        "raw_shards": [str(path) for path in raw_paths],
        "by_instrument": {
            instrument: {
                "valid": sum(
                    completion_counts[(teacher.id, instrument, True)]
                    for teacher in config.labeling.models
                ),
                "invalid": sum(
                    completion_counts[(teacher.id, instrument, False)]
                    for teacher in config.labeling.models
                ),
            }
            for instrument in config.labeling.instruments
        },
        "by_teacher": {
            teacher.id: {
                "model": teacher.model,
                "valid": sum(
                    completion_counts[(teacher.id, instrument, True)]
                    for instrument in config.labeling.instruments
                ),
                "invalid": sum(
                    completion_counts[(teacher.id, instrument, False)]
                    for instrument in config.labeling.instruments
                ),
            }
            for teacher in config.labeling.models
        },
        "teacher_agreement": _teacher_agreement(labels, config.labeling.models),
        "dimensions": summaries,
    }
    atomic_json(artifacts.label_quality, quality)
    manifest = stage_manifest(
        config,
        "labels",
        inputs={
            "raw_label_shards": {
                str(path): sha256_file(path) for path in raw_paths
            },
            "work_manifests_sha256": {
                teacher_id: sha256_file(path)
                for teacher_id, path in work_manifests.items()
            },
            "dataset_manifest_sha256": sha256_file(artifacts.dataset_manifest),
        },
        details=quality,
    )
    manifest["artifacts"] = {
        "labels_sha256": sha256_file(artifacts.labels),
        "quality_sha256": sha256_file(artifacts.label_quality),
    }
    atomic_json(artifacts.labels.with_suffix(".manifest.json"), manifest)
    return labels, quality


def require_complete_labels(config, artifacts) -> dict:
    """Validate parsed-label lineage and confirm every planned task is valid."""
    from syco.linear_probe.artifacts import require_manifest

    manifest = require_manifest(
        artifacts.labels.with_suffix(".manifest.json"), config, "labels"
    )
    artifacts_hashes = manifest.get("artifacts") or {}
    if sha256_file(artifacts.labels) != artifacts_hashes.get("labels_sha256"):
        raise ValueError("parsed label artifact hash mismatch")
    if sha256_file(artifacts.label_quality) != artifacts_hashes.get("quality_sha256"):
        raise ValueError("label quality artifact hash mismatch")
    quality = manifest.get("details") or {}
    expected = int(quality.get("expected_completions", -1))
    valid = int(quality.get("valid_completions", -1))
    invalid = int(quality.get("invalid_completions", -1))
    missing = int(quality.get("missing_completions", -1))
    if expected < 0 or valid != expected or invalid != 0 or missing != 0:
        raise RuntimeError(
            "parsed labels are not complete and strictly valid: "
            f"expected={expected}, valid={valid}, invalid={invalid}, missing={missing}"
        )
    return manifest
