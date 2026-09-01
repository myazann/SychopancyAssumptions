"""Causal residual-stream steering and deterministic Yes/No option scores."""

from __future__ import annotations

import datetime as dt
import hashlib
import math

import numpy as np
import pandas as pd

from syco.grid import stable_sample
from syco.linear_probe.artifacts import (
    append_jsonl,
    atomic_json,
    cell_id,
    design_unit_id,
    read_jsonl,
    require_manifest,
    sha256_file,
    stage_manifest,
)
from syco.linear_probe.dataset import load_frozen_cells
from syco.linear_probe.modeling import (
    load_target_model,
    load_tokenizer,
    pick_input_device,
    resolve_decoder_blocks,
    target_fingerprint,
)
from syco.linear_probe.prompts import render_target_prompt
from syco.linear_probe.training import load_probe


class SteeringHook:
    """Add a direction to every token at one decoder block output.

    ``strength > 0`` always means toward the higher probe score. At zero the
    hook returns the original object without arithmetic, enabling an exact
    unhooked-equality check.
    """

    def __init__(self, block, direction):
        import torch

        tensor = torch.as_tensor(direction, dtype=torch.float32)
        norm = tensor.norm()
        if not torch.isfinite(norm) or float(norm) <= 1e-12:
            raise ValueError("steering direction is zero or nonfinite")
        self.direction = tensor / norm
        self.strength = 0.0
        self.handle = block.register_forward_hook(self._apply)

    def _apply(self, module, inputs, output):
        if self.strength == 0.0:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.shape[-1] != len(self.direction):
            raise RuntimeError(
                f"direction width {len(self.direction)} != block width "
                f"{hidden.shape[-1]}"
            )
        delta = (self.direction.to(device=hidden.device, dtype=hidden.dtype)
                 * float(self.strength))
        moved = hidden + delta
        return (moved, *output[1:]) if isinstance(output, tuple) else moved

    def close(self):
        self.handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _option_token_ids(tokenizer, answer: str) -> list[int]:
    ids = tokenizer.encode(answer, add_special_tokens=False)
    if not ids:
        raise ValueError(f"answer candidate {answer!r} tokenizes to nothing")
    return [int(value) for value in ids]


def score_options(model, tokenizer, prompts: list[str], answer_yes: str,
                  answer_no: str, input_device, *, max_length: int,
                  batch_size: int) -> list[dict]:
    """Conditional sequence log likelihood for complete Yes and No strings."""
    import torch

    if getattr(tokenizer, "padding_side", None) != "left":
        raise ValueError("option scoring requires tokenizer.padding_side='left'")
    candidates = (("yes", answer_yes), ("no", answer_no))
    prompt_ids = [tokenizer.encode(text, add_special_tokens=False) for text in prompts]
    output = [{"logp_yes": None, "logp_no": None} for _ in prompts]

    examples = []
    for prompt_index, (prompt, prefix) in enumerate(zip(prompts, prompt_ids)):
        if not prefix:
            raise ValueError("rendered target prompt tokenizes to nothing")
        for name, answer in candidates:
            combined = tokenizer.encode(prompt + answer, add_special_tokens=False)
            if combined[:len(prefix)] != prefix:
                raise RuntimeError(
                    f"tokenization at the generation boundary changes the prompt "
                    f"for candidate {answer!r}; choose an answer spelling/leading "
                    "space that preserves the rendered prompt token prefix"
                )
            suffix = combined[len(prefix):]
            if not suffix:
                raise RuntimeError(f"candidate {answer!r} adds no answer tokens")
            if len(prefix) + len(suffix) > max_length:
                raise RuntimeError(
                    f"scoring prompt {prompt_index} needs {len(prefix)+len(suffix)} "
                    f"tokens, over max_length={max_length}; no truncation applied"
                )
            examples.append((prompt_index, name, prefix, suffix))

    sequence_batch = max(1, batch_size * 2)
    with torch.inference_mode():
        for start in range(0, len(examples), sequence_batch):
            batch = examples[start:start + sequence_batch]
            features = [{"input_ids": prefix + suffix}
                        for _, _, prefix, suffix in batch]
            encoded = tokenizer.pad(features, padding=True, return_tensors="pt")
            encoded = {key: value.to(input_device) for key, value in encoded.items()}
            logits = model(**encoded, use_cache=False, return_dict=True).logits
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            padded_width = encoded["input_ids"].shape[1]
            for row, (prompt_index, name, prefix, suffix) in enumerate(batch):
                pad = padded_width - len(prefix) - len(suffix)
                answer_start = pad + len(prefix)
                total = 0.0
                for offset, token_id in enumerate(suffix):
                    logit_position = answer_start + offset - 1
                    total += float(log_probs[row, logit_position, token_id].item())
                output[prompt_index][f"logp_{name}"] = total
    for result in output:
        result["logit_no"] = result["logp_no"] - result["logp_yes"]
        result["log_candidate_mass"] = float(np.logaddexp(
            result["logp_yes"], result["logp_no"]
        ))
        result["candidate_mass"] = math.exp(min(
            result["log_candidate_mass"], 0.0
        ))
        value = result["logit_no"]
        if value >= 0:
            result["p_no"] = 1.0 / (1.0 + math.exp(-min(value, 700)))
        else:
            exp_value = math.exp(max(value, -700))
            result["p_no"] = exp_value / (1.0 + exp_value)
    return output


def _score_key(config, cell, dimension, alpha, direction_kind):
    return "|".join((
        config.stage_digest("steering"), cell_id(cell), dimension, direction_kind,
        format(float(alpha), ".12g"),
    ))


def _done_keys(path) -> set[str]:
    return {
        row["score_key"] for row in read_jsonl(path)
        if row.get("score_key") and not row.get("error")
    }


def _random_direction(width: int, seed: int, dimension: str,
                      control_index: int) -> np.ndarray:
    digest = int(hashlib.sha256(
        f"{seed}|{dimension}|{control_index}".encode()
    ).hexdigest()[:16], 16)
    rng = np.random.default_rng(digest)
    direction = rng.standard_normal(width).astype(np.float32)
    return direction / np.linalg.norm(direction)


def _records(config, cells, dataset_rows, scores, *, dimension, alpha,
             applied_strength, block_index, direction_kind):
    timestamp = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    records = []
    for cell, row, score in zip(cells, dataset_rows.itertuples(index=False), scores):
        records.append({
            "schema_version": 1,
            "config_digest": config.stage_digest("steering"),
            "pipeline_digest": config.digest,
            "score_key": _score_key(config, cell, dimension, alpha, direction_kind),
            "cell_id": cell_id(cell),
            "design_unit_id": design_unit_id(cell),
            "row_index": int(row.row_index),
            "persona_type": cell.persona.persona_type,
            "persona_id": cell.persona.persona_id,
            "prompt_type": cell.prompt.prompt_type,
            "prompt_id": cell.prompt.prompt_id,
            "rep": cell.rep,
            "split": row.split,
            "dimension": dimension,
            "direction_kind": direction_kind,
            "alpha": float(alpha),
            "applied_strength": float(applied_strength),
            "block_index": block_index,
            **score,
            "error": "",
            "timestamp": timestamp,
        })
    return records


def _score_missing(config, artifacts, model, tokenizer, input_device, cells,
                   table, *, dimension, alpha, applied_strength, block_index,
                   direction_kind, done, prompt_lookup):
    missing = [index for index, cell in enumerate(cells)
               if _score_key(config, cell, dimension, alpha, direction_kind) not in done]
    if not missing:
        return 0
    written = 0
    prompt_batch = max(1, config.target.batch_size)
    for start in range(0, len(missing), prompt_batch):
        indices = missing[start:start + prompt_batch]
        batch_cells = [cells[index] for index in indices]
        batch_rows = table.iloc[indices]
        prompts = [prompt_lookup[cell_id(cell)] for cell in batch_cells]
        scores = score_options(
            model, tokenizer, prompts,
            config.steering.answer_yes, config.steering.answer_no,
            input_device, max_length=config.target.max_length,
            batch_size=config.target.batch_size,
        )
        records = _records(
            config, batch_cells, batch_rows, scores, dimension=dimension,
            alpha=alpha, applied_strength=applied_strength,
            block_index=block_index, direction_kind=direction_kind,
        )
        append_jsonl(artifacts.steering, records)
        done.update(row["score_key"] for row in records)
        written += len(records)
    return written


def run_steering(config, artifacts, *, allow_weak_probe: bool = False) -> dict:
    probe_manifest = require_manifest(
        artifacts.probes / "manifest.json", config, "probes"
    )
    cells, table = load_frozen_cells(config, artifacts)
    if config.steering.partition != "all":
        keep = table.split == config.steering.partition
        table = table.loc[keep].reset_index(drop=True)
        cells = [cell for cell, selected in zip(cells, keep) if selected]
    if not cells:
        raise RuntimeError(
            f"no cells in steering partition {config.steering.partition!r}"
        )
    units = list(dict.fromkeys(table.design_unit_id.astype(str)))
    selected_units = set(stable_sample(
        units, config.steering.max_design_units, config.steering.seed,
        "steering-units",
    ))
    keep = table.design_unit_id.astype(str).isin(selected_units)
    table = table.loc[keep].reset_index(drop=True)
    cells = [cell for cell, selected in zip(cells, keep) if selected]

    tokenizer = load_tokenizer(config.target)
    model = load_target_model(config.target)
    blocks, block_path = resolve_decoder_blocks(model)
    expected_fp = probe_manifest["details"]["target_fingerprint"]
    current_fp = target_fingerprint(
        model, tokenizer, config.target, block_path,
        expected_fp["candidate_block_indices"],
    )
    if current_fp["fingerprint"] != expected_fp["fingerprint"]:
        raise RuntimeError(
            "target checkpoint/tokenizer/block fingerprint differs from activation "
            "extraction; refusing to steer with incompatible probe weights"
        )
    activation_manifest = require_manifest(
        artifacts.activations / "manifest.json", config, "activations"
    )
    activation_manifest_hash = sha256_file(
        artifacts.activations / "manifest.json"
    )
    if probe_manifest.get("inputs", {}).get(
            "activations_manifest_sha256") != activation_manifest_hash:
        raise ValueError(
            "probe manifest does not refer to the current activation manifest"
        )
    if sha256_file(artifacts.activation_rows) != (
            activation_manifest.get("artifacts") or {}).get("rows_sha256"):
        raise ValueError("activation row metadata hash mismatch")
    activation_rows = pd.read_parquet(
        artifacts.activation_rows,
        columns=["cell_id", "target_prompt_digest"],
    )
    expected_prompt_digests = dict(zip(
        activation_rows.cell_id.astype(str),
        activation_rows.target_prompt_digest.astype(str),
    ))
    prompt_lookup = {}
    for cell in cells:
        key = cell_id(cell)
        prompt = render_target_prompt(tokenizer, cell, config.target)
        digest = hashlib.sha256(prompt.encode()).hexdigest()[:20]
        if expected_prompt_digests.get(key) != digest:
            raise RuntimeError(
                f"steering prompt for {key} does not match activation extraction"
            )
        prompt_lookup[key] = prompt
    probes_manifest_hash = sha256_file(artifacts.probes / "manifest.json")
    dataset_manifest_hash = sha256_file(artifacts.dataset_manifest)
    selected_cells_digest = hashlib.sha256(
        "\n".join(cell_id(cell) for cell in cells).encode()
    ).hexdigest()
    work_manifest_path = artifacts.steering.parent / "work_manifest.json"
    work_details = {
        "probes_manifest_sha256": probes_manifest_hash,
        "dataset_manifest_sha256": dataset_manifest_hash,
        "target_fingerprint": current_fp,
        "selected_cells_digest": selected_cells_digest,
        "activations_manifest_sha256": activation_manifest_hash,
        "representation_input_digest": activation_manifest["details"][
            "representation_input_digest"
        ],
    }
    if work_manifest_path.is_file():
        work_manifest = require_manifest(
            work_manifest_path, config, "steering_work"
        )
        if work_manifest.get("details") != work_details:
            raise RuntimeError(
                "partial steering scores were produced by different probes, "
                "target weights, or selected cells; use a separate output directory"
            )
    else:
        if artifacts.steering.is_file():
            raise RuntimeError(
                "steering scores exist without a provenance work manifest; refusing "
                "to mix them with this run"
            )
        atomic_json(
            work_manifest_path,
            stage_manifest(config, "steering_work", details=work_details),
        )
    input_device = pick_input_device(model)
    done = _done_keys(artifacts.steering)
    written = 0
    answer_token_ids = {
        "yes": _option_token_ids(tokenizer, config.steering.answer_yes),
        "no": _option_token_ids(tokenizer, config.steering.answer_no),
    }
    try:
        # One unhooked alpha-zero baseline is shared across every dimension.
        written += _score_missing(
            config, artifacts, model, tokenizer, input_device, cells, table,
            dimension="__baseline__", alpha=0.0, applied_strength=0.0,
            block_index=None, direction_kind="unsteered", done=done,
            prompt_lookup=prompt_lookup,
        )
        for dimension in config.steering.dimensions:
            weights, _metadata = load_probe(
                config, artifacts, dimension, allow_weak=allow_weak_probe
            )
            block_index = int(weights["block_index"])
            if not 0 <= block_index < len(blocks):
                raise RuntimeError(
                    f"{dimension} requests block {block_index} in {len(blocks)} blocks"
                )
            directions = [("probe", weights["unit_direction"])]
            directions.extend(
                (f"random_{index}", _random_direction(
                    len(weights["unit_direction"]), config.steering.seed,
                    dimension, index,
                ))
                for index in range(config.steering.random_control_count)
            )
            for direction_kind, direction in directions:
                probe_cells = cells[:min(len(cells), config.target.batch_size)]
                prompts = [prompt_lookup[cell_id(cell)] for cell in probe_cells]
                unhooked_scores = score_options(
                    model, tokenizer, prompts,
                    config.steering.answer_yes, config.steering.answer_no,
                    input_device, max_length=config.target.max_length,
                    batch_size=config.target.batch_size,
                )
                with SteeringHook(blocks[block_index], direction) as hook:
                    # A registered zero hook must reproduce unhooked scores exactly.
                    zero_scores = score_options(
                        model, tokenizer, prompts,
                        config.steering.answer_yes, config.steering.answer_no,
                        input_device, max_length=config.target.max_length,
                        batch_size=config.target.batch_size,
                    )
                    for cell, score, unhooked in zip(
                            probe_cells, zero_scores, unhooked_scores):
                        if score["logp_yes"] != unhooked["logp_yes"] or \
                                score["logp_no"] != unhooked["logp_no"]:
                            raise RuntimeError(
                                f"alpha=0 hook changed logits for {cell_id(cell)}"
                            )
                    for alpha in config.steering.alphas:
                        if alpha == 0:
                            continue
                        scale = (float(weights["projection_std"])
                                 if config.steering.scale == "projected_std" else 1.0)
                        hook.strength = float(alpha) * scale
                        written += _score_missing(
                            config, artifacts, model, tokenizer, input_device,
                            cells, table, dimension=dimension, alpha=alpha,
                            applied_strength=hook.strength,
                            block_index=block_index,
                            direction_kind=direction_kind, done=done,
                            prompt_lookup=prompt_lookup,
                        )
    finally:
        del model
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    manifest = stage_manifest(
        config,
        "steering",
        inputs={
            "probes_manifest_sha256": probes_manifest_hash,
            "dataset_manifest_sha256": dataset_manifest_hash,
        },
        details={
            "rows_written_this_run": written,
            "selected_cells": len(cells),
            "selected_design_units": len(selected_units),
            "partition": config.steering.partition,
            "answer_token_ids": answer_token_ids,
            "target_fingerprint": current_fp,
            "zero_hook_equals_unhooked": True,
            "positive_alpha_semantics": "toward higher labelled assumption score",
        },
    )
    manifest["artifacts"] = {
        "scores_sha256": sha256_file(artifacts.steering),
    }
    atomic_json(artifacts.steering.with_suffix(".manifest.json"), manifest)
    return manifest
