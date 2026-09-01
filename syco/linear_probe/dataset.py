"""Freeze paired examples and leakage-safe partitions before labeling."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

import pandas as pd

from syco import paths
from syco.data import Persona, Prompt
from syco.grid import Cell
from syco.linear_probe.artifacts import (
    atomic_json,
    build_design,
    cell_id,
    coordinate,
    design_unit_id,
    prompt_digest,
    require_manifest,
    resolve_input,
    sha256_file,
    stable_group_partitions,
    stage_manifest,
)
from syco.linear_probe.prompts import build_label_prompt, deployment_messages


def _partition_groups(values, split, tag: str) -> dict[str, str]:
    return stable_group_partitions(values, split, tag)


def assign_splits(rows: pd.DataFrame, split) -> pd.DataFrame:
    """Assign entire persona and dilemma groups, never individual framings."""
    out = rows.copy()
    persona_values = (
        out.loc[out["persona_id"] != "none", "persona_id"]
        if split.group_by == "two_axis" else out["persona_id"]
    )
    persona = _partition_groups(persona_values, split, "persona_id")
    prompt = _partition_groups(out["prompt_id"], split, "prompt_id")
    out["persona_partition"] = out["persona_id"].map(persona)
    out["prompt_partition"] = out["prompt_id"].map(prompt)

    if split.group_by == "two_axis":
        same = out["persona_partition"] == out["prompt_partition"]
        out["split"] = (
            out["persona_partition"].where(same, "cross_" +
                out["persona_partition"] + "_" + out["prompt_partition"])
        )
        # A no-persona control has no identity axis to leak. Assign it solely
        # by dilemma so controls are represented in every primary partition.
        control = out["persona_id"] == "none"
        out.loc[control, "persona_partition"] = "not_applicable"
        out.loc[control, "split"] = out.loc[control, "prompt_partition"]
    elif split.group_by == "persona_id":
        out["split"] = out["persona_partition"]
    elif split.group_by == "prompt_id":
        out["split"] = out["prompt_partition"]
    else:
        groups = _partition_groups(out["design_unit_id"], split, "cell")
        out["split"] = out["design_unit_id"].map(groups)
    return out


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def freeze_dataset(config, artifacts) -> pd.DataFrame:
    """Create immutable, deduplicated data snapshots and a coordinate table."""
    if artifacts.dataset_manifest.exists():
        require_manifest(artifacts.dataset_manifest, config, "dataset")
        for path in (artifacts.dataset, artifacts.dataset_personas,
                     artifacts.dataset_prompts, artifacts.dataset_demographics):
            if not path.is_file():
                raise FileNotFoundError(f"dataset manifest exists but {path} is missing")
        return pd.read_parquet(artifacts.dataset)

    cells, diagnostics, persona_source, prompt_source = build_design(config)
    if not cells:
        raise ValueError("the configured design contains no cells")

    cell_rows = []
    for index, cell in enumerate(cells):
        row = coordinate(cell)
        row.update(
            row_index=index,
            cell_id=cell_id(cell),
            design_unit_id=design_unit_id(cell),
            persona_turns=cell.persona.n_turns,
            persona_recovered=cell.persona.recovered,
            semantic_prompt_digest=prompt_digest(
                deployment_messages(cell, config.target.answer_instruction),
                config.target.system_prompt,
            ),
        )
        for instrument in config.labeling.instruments:
            text = build_label_prompt(
                instrument, cell.persona.messages, cell.prompt.text
            )
            row[f"label_prompt_digest_{instrument}"] = hashlib.sha256(
                text.encode()
            ).hexdigest()[:20]
        cell_rows.append(row)
    table = assign_splits(pd.DataFrame(cell_rows), config.training.split)
    generated_cells = len(table)
    excluded_cross_axis = 0
    if (config.training.split.group_by == "two_axis"
            and not config.design.include_cross_axis):
        primary = table["split"].isin({"train", "validation", "test"})
        excluded_cross_axis = int((~primary).sum())
        selected_positions = table.loc[primary, "row_index"].astype(int).tolist()
        cells = [cells[position] for position in selected_positions]
        table = table.loc[primary].reset_index(drop=True)
        table["row_index"] = range(len(table))

    persona_rows = {}
    prompt_rows = {}
    for cell in cells:
        pkey = (cell.persona.persona_type, cell.persona.persona_id)
        persona_rows[pkey] = {
            "persona_type": pkey[0],
            "persona_id": pkey[1],
            "messages_json": json.dumps(
                list(cell.persona.messages), ensure_ascii=False, separators=(",", ":")
            ),
            "recovered": cell.persona.recovered,
            "n_turns": cell.persona.n_turns,
        }
        qkey = (cell.prompt.prompt_type, cell.prompt.prompt_id)
        prompt_rows[qkey] = {
            "prompt_type": qkey[0],
            "prompt_id": qkey[1],
            "text": cell.prompt.text,
        }

    demographics_source = resolve_input(
        config.design.demographics_path, paths.DEMOGRAPHICS_PATH
    )
    if not demographics_source.is_file():
        raise FileNotFoundError(
            f"demographic/vulnerability table not found: {demographics_source}"
        )
    demographics = pd.read_csv(demographics_source)
    if "uuid" not in demographics:
        raise ValueError("demographic/vulnerability table must contain uuid")
    demographics["uuid"] = demographics["uuid"].astype(str)
    if demographics.uuid.duplicated().any():
        raise ValueError("demographic/vulnerability table contains duplicate uuid")
    selected_ids = {
        str(cell.persona.persona_id)
        for cell in cells
        if cell.persona.persona_id != "none"
    }
    missing_demographics = sorted(selected_ids - set(demographics.uuid))
    if missing_demographics:
        raise ValueError(
            f"demographic/vulnerability table is missing {len(missing_demographics)} "
            f"selected persona IDs; first entries: {missing_demographics[:5]}"
        )
    demographics = demographics[demographics.uuid.isin(selected_ids)].copy()
    demographics.insert(0, "persona_id", demographics.pop("uuid"))
    demographics = demographics.sort_values("persona_id").reset_index(drop=True)

    _atomic_parquet(table, artifacts.dataset)
    _atomic_parquet(pd.DataFrame(persona_rows.values()), artifacts.dataset_personas)
    _atomic_parquet(pd.DataFrame(prompt_rows.values()), artifacts.dataset_prompts)
    _atomic_parquet(demographics, artifacts.dataset_demographics)
    details = {
        "cells": len(table),
        "candidate_cells_before_split_filter": generated_cells,
        "excluded_cross_axis_cells": excluded_cross_axis,
        "design_units": int(table["design_unit_id"].nunique()),
        "people": int(table.loc[table.persona_id != "none", "persona_id"].nunique()),
        "dilemmas": int(table["prompt_id"].nunique()),
        "facets": int(table["persona_type"].nunique()),
        "split_cells": dict(Counter(table["split"])),
        "unparseable_persona_rows": int((~diagnostics["usable"]).sum()),
        "demographic_rows": len(demographics),
        "demographic_columns": list(demographics.columns),
    }
    manifest = stage_manifest(
        config,
        "dataset",
        inputs={
            "personas": str(persona_source.resolve()),
            "personas_sha256": sha256_file(persona_source),
            "prompts": str(prompt_source.resolve()),
            "prompts_sha256": sha256_file(prompt_source),
            "demographics": str(demographics_source.resolve()),
            "demographics_sha256": sha256_file(demographics_source),
        },
        details=details,
    )
    manifest["artifacts"] = {
        "cells_sha256": sha256_file(artifacts.dataset),
        "personas_sha256": sha256_file(artifacts.dataset_personas),
        "prompts_sha256": sha256_file(artifacts.dataset_prompts),
        "demographics_sha256": sha256_file(artifacts.dataset_demographics),
    }
    atomic_json(artifacts.dataset_manifest, manifest)
    return table


def load_frozen_cells(config, artifacts) -> tuple[list[Cell], pd.DataFrame]:
    """Load cells from the frozen snapshot, independent of mutable base files."""
    manifest = require_manifest(artifacts.dataset_manifest, config, "dataset")
    expected = manifest.get("artifacts") or {}
    for name, path in (
        ("cells", artifacts.dataset),
        ("personas", artifacts.dataset_personas),
        ("prompts", artifacts.dataset_prompts),
        ("demographics", artifacts.dataset_demographics),
    ):
        if sha256_file(path) != expected.get(f"{name}_sha256"):
            raise ValueError(f"frozen dataset artifact changed: {path}")

    table = pd.read_parquet(artifacts.dataset).sort_values("row_index")
    personas = {}
    for row in pd.read_parquet(artifacts.dataset_personas).itertuples(index=False):
        messages = tuple(json.loads(row.messages_json))
        personas[(str(row.persona_type), str(row.persona_id))] = Persona(
            str(row.persona_id), str(row.persona_type), messages,
            bool(row.recovered), int(row.n_turns),
        )
    prompts = {}
    for row in pd.read_parquet(artifacts.dataset_prompts).itertuples(index=False):
        prompts[(str(row.prompt_type), str(row.prompt_id))] = Prompt(
            str(row.prompt_id), str(row.prompt_type), str(row.text)
        )

    cells = []
    for row in table.itertuples(index=False):
        cell = Cell(
            persona=personas[(str(row.persona_type), str(row.persona_id))],
            prompt=prompts[(str(row.prompt_type), str(row.prompt_id))],
            rep=int(row.rep),
        )
        if cell_id(cell) != row.cell_id:
            raise ValueError(f"frozen cell identity mismatch at row {row.row_index}")
        cells.append(cell)
    return cells, table.reset_index(drop=True)


def summarize_dataset(table: pd.DataFrame) -> str:
    counts = table["split"].value_counts().to_dict()
    primary = {key: counts.get(key, 0) for key in ("train", "validation", "test")}
    excluded = len(table) - sum(primary.values())
    return (
        f"{len(table):,} framing-specific cells / "
        f"{table['design_unit_id'].nunique():,} paired units; "
        f"train={primary['train']:,}, validation={primary['validation']:,}, "
        f"test={primary['test']:,}, cross-axis diagnostic={excluded:,}"
    )
