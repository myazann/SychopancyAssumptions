"""Command line stages for the fresh-label linear-probe study."""

from __future__ import annotations

import argparse
import time

import pandas as pd

from syco.linear_probe.artifacts import (
    ArtifactPaths,
    build_design,
    cell_id,
    design_unit_id,
    paths_for,
)
from syco.linear_probe.config import DEFAULT_CONFIG_PATH, load_config
from syco.linear_probe.dataset import assign_splits, freeze_dataset, summarize_dataset

STAGES = ("plan", "freeze", "label", "parse-labels", "extract", "train",
          "steer", "evaluate", "status")

# Provisional project-local calibration. Replace these with the 200-call
# label-only benchmark on the allocated node before scheduling the full array.
LABEL_SECONDS_PER_CALL = (3.2, 4.5)
LEGACY_LABEL_AND_RESPONSE_SECONDS = 6.95
GEMMA27_LEGACY_SECONDS_PER_CALL = (18.05, 18.89)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m syco linear-probe",
        description=(
            "Fresh multi-teacher labels -> target activations -> Ridge probes "
            "-> steering"
        ),
    )
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument(
        "--config", default=str(DEFAULT_CONFIG_PATH),
        help="linear-probe YAML configuration",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="override the configuration's artifact root",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="label tasks or extraction rows to process (pilot/debug only)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="label stage only: strict synthetic completions in a dry_run namespace",
    )
    parser.add_argument(
        "--allow-weak-probe", action="store_true",
        help="steer even if a probe failed its validation/distribution gate",
    )
    parser.add_argument(
        "--allow-partial", action="store_true",
        help="parse-labels: do not fail solely because a pilot/shard is incomplete",
    )
    parser.add_argument(
        "--num-shards", type=int, default=1,
        help="label stage: total deterministic GPU shards (default: 1)",
    )
    parser.add_argument(
        "--shard-index", type=int, default=0,
        help="label stage: zero-based shard index (default: 0)",
    )
    parser.add_argument(
        "--teacher", default=None,
        help="label stage: configured teacher ID (required with multiple teachers)",
    )
    return parser


def _in_memory_plan(config) -> pd.DataFrame:
    cells, _, _, _ = build_design(config)
    rows = []
    for index, cell in enumerate(cells):
        rows.append({
            "row_index": index,
            "cell_id": cell_id(cell),
            "design_unit_id": design_unit_id(cell),
            "persona_id": cell.persona.persona_id,
            "prompt_id": cell.prompt.prompt_id,
            "persona_type": cell.persona.persona_type,
            "prompt_type": cell.prompt.prompt_type,
        })
    table = assign_splits(pd.DataFrame(rows), config.training.split)
    excluded = 0
    if (config.training.split.group_by == "two_axis"
            and not config.design.include_cross_axis):
        keep = table.split.isin({"train", "validation", "test"})
        excluded = int((~keep).sum())
        table = table.loc[keep].reset_index(drop=True)
        table["row_index"] = range(len(table))
    table.attrs["excluded_cross_axis"] = excluded
    return table


def _plan(config) -> int:
    table = _in_memory_plan(config)
    calls_per_teacher = (len(table) * len(config.labeling.instruments)
                         * config.labeling.replicates)
    label_calls = calls_per_teacher * len(config.labeling.models)
    print(f"config:      {config.source_path}")
    print(f"digest:      {config.digest}")
    print("stage IDs:   " + ", ".join(
        f"{stage}={config.stage_digest(stage)}"
        for stage in ("dataset", "labels", "activations", "probes", "steering")
    ))
    print(f"artifacts:   {config.root}")
    print(summarize_dataset(table))
    if table.attrs.get("excluded_cross_axis"):
        print(
            f"excluded:    {table.attrs['excluded_cross_axis']:,} cross-axis cells "
            "before labeling (set design.include_cross_axis=true to retain them)"
        )
    print(
        f"label calls: {label_calls:,} = {len(table):,} cells x "
        f"{len(config.labeling.instruments)} instruments x "
        f"{config.labeling.replicates} replicate(s) x "
        f"{len(config.labeling.models)} teacher(s)"
    )
    print("teachers:     " + ", ".join(
        f"{model.id}={model.model}" for model in config.labeling.models
    ))
    print(
        f"sampling:     temperature={config.labeling.temperature}; strict JSON; "
        f"max attempts={config.labeling.max_attempts}"
    )
    low_hours, high_hours = (
        calls_per_teacher * seconds / 3600 for seconds in LABEL_SECONDS_PER_CALL
    )
    legacy_hours = (
        calls_per_teacher * LEGACY_LABEL_AND_RESPONSE_SECONDS / 3600
    )
    print(
        f"Qwen time:   provisional {low_hours:.1f}-{high_hours:.1f} GPU-h "
        f"({low_hours / 4:.1f}-{high_hours / 4:.1f} h ideal wall time on four "
        f"fixed shards); legacy labels+reply upper baseline {legacy_hours:.1f} GPU-h"
    )
    gemma_low, gemma_high = (
        calls_per_teacher * seconds / 3600
        for seconds in GEMMA27_LEGACY_SECONDS_PER_CALL
    )
    if any(model.model == "Gemma3-27B" for model in config.labeling.models):
        print(
            f"Gemma time: conservative prior labels+reply baseline "
            f"{gemma_low:.1f}-{gemma_high:.1f} GPU-h "
            f"({gemma_low / 10:.1f}-{gemma_high / 10:.1f} h on ten shards); "
            "replace with the two-teacher pilot benchmark"
        )
    target_name = config.target.hf_ref or "UNSELECTED (labeling can proceed)"
    print(
        f"target:      {target_name} ({config.target.dtype}); "
        f"pooling={config.target.pooling}; max_length={config.target.max_length}/"
        f"{config.target.overlength}"
    )
    print(f"blocks:      {list(config.target.layers.explicit) or list(config.target.layers.fractions)}")
    print(
        "before the full label stage: run a real --limit 200 benchmark for each "
        "teacher, inspect raw.teacher-*.jsonl, parse labels, and human-audit a "
        "stratified sample"
    )
    return 0


def _status(config, artifacts: ArtifactPaths) -> int:
    from syco.linear_probe.labels import raw_label_paths

    files = (
        ("dataset", artifacts.dataset_manifest),
        ("raw labels", artifacts.raw_labels if artifacts.raw_labels.exists()
         else (raw_label_paths(artifacts)[0] if raw_label_paths(artifacts)
               else artifacts.raw_labels)),
        ("parsed labels", artifacts.labels),
        ("activations", artifacts.activations / "manifest.json"),
        ("probes", artifacts.probes / "manifest.json"),
        ("steering", artifacts.steering),
        ("evaluation", artifacts.evaluation / "manifest.json"),
    )
    print(f"pipeline digest: {config.digest}")
    for name, path in files:
        print(f"{name:<14} {'ready' if path.exists() else 'missing':<7} {path}")
    return 0


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.dry_run and args.stage not in {"label", "parse-labels", "status"}:
        raise ValueError(
            "--dry-run selects the synthetic label namespace for label, "
            "parse-labels, or status"
        )
    if args.limit is not None and args.stage not in {"label", "extract"}:
        raise ValueError("--limit applies only to label or extract")
    if args.stage != "label" and (args.num_shards != 1 or args.shard_index != 0):
        raise ValueError("--num-shards/--shard-index apply only to label")
    if args.stage != "label" and args.teacher is not None:
        raise ValueError("--teacher applies only to label")
    if args.stage != "parse-labels" and args.allow_partial:
        raise ValueError("--allow-partial applies only to parse-labels")
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("require 0 <= --shard-index < --num-shards")
    config = load_config(args.config).with_output_dir(args.output_dir)
    artifacts = paths_for(config, dry_run=args.dry_run)
    if args.stage == "plan":
        return _plan(config)
    if args.stage == "status":
        return _status(config, artifacts)
    if args.stage == "freeze":
        table = freeze_dataset(config, artifacts)
        print(summarize_dataset(table))
        print(f"wrote {artifacts.dataset}")
        return 0
    if args.stage == "label":
        from syco.linear_probe.labels import run_labeling
        started = time.monotonic()
        result = run_labeling(
            config, artifacts, dry_run=args.dry_run, limit=args.limit,
            shard_index=args.shard_index, num_shards=args.num_shards,
            teacher_id=args.teacher,
        )
        elapsed = time.monotonic() - started
        print(
            f"labeling[{result['teacher_id']}]: planned={result['planned']:,}, "
            f"attempts={result['written']:,}, "
            f"valid={result['valid']:,}, invalid={result['invalid']:,}, "
            f"elapsed={elapsed / 60:.1f} min"
        )
        if result["written"]:
            print(f"empirical rate: {elapsed / result['written']:.3f} s/attempt")
        print(f"raw labels: {result['raw_path']}")
        return 0
    if args.stage == "parse-labels":
        from syco.linear_probe.labels import parse_labels
        labels, quality = parse_labels(config, artifacts)
        print(
            f"{quality['valid_completions']:,} valid / "
            f"{quality['canonical_completions']:,} canonical completions -> "
            f"{len(labels):,} dimension rows"
        )
        print(f"wrote {artifacts.labels}")
        return 1 if (quality["invalid_completions"]
                     or (quality.get("missing_completions")
                         and not args.allow_partial)) else 0
    if args.stage == "extract":
        from syco.linear_probe.activations import extract_activations
        manifest = extract_activations(config, artifacts, limit=args.limit)
        print(json_summary(manifest["details"]))
        print(f"wrote {artifacts.activations}")
        return 0
    if args.stage == "train":
        from syco.linear_probe.training import train_probes
        manifest = train_probes(config, artifacts)
        print(json_summary(manifest["details"]))
        print(f"wrote {artifacts.probes}")
        return 0
    if args.stage == "steer":
        from syco.linear_probe.steering import run_steering
        manifest = run_steering(
            config, artifacts, allow_weak_probe=args.allow_weak_probe
        )
        print(json_summary(manifest["details"]))
        print(f"wrote {artifacts.steering}")
        return 0
    if args.stage == "evaluate":
        from syco.linear_probe.evaluation import evaluate_steering
        manifest = evaluate_steering(config, artifacts)
        print(json_summary(manifest["details"]))
        print(f"wrote {artifacts.evaluation}")
        return 0
    raise AssertionError(args.stage)


def json_summary(value) -> str:
    import json
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
