"""One-pass, multi-block pooled activation extraction with resumable shards."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from syco.linear_probe.artifacts import (
    atomic_json,
    prompt_digest,
    require_manifest,
    sha256_file,
    stage_manifest,
)
from syco.linear_probe.dataset import _atomic_parquet, load_frozen_cells
from syco.linear_probe.modeling import (
    candidate_block_indices,
    load_target_model,
    load_tokenizer,
    pick_input_device,
    resolve_decoder_blocks,
    target_fingerprint,
)
from syco.linear_probe.prompts import (
    deployment_messages,
    deployment_user_text,
    render_target_prompt,
)


def pool_hidden(hidden, mask, mode: str):
    """Pool ``[batch, tokens, width]`` using a padding/span-safe mask."""
    import torch

    if hidden.ndim != 3 or mask.ndim != 2 or hidden.shape[:2] != mask.shape:
        raise ValueError(
            f"hidden/mask shape mismatch: {tuple(hidden.shape)} vs {tuple(mask.shape)}"
        )
    valid = mask.to(device=hidden.device, dtype=torch.bool)
    if not bool(valid.any(dim=1).all()):
        raise ValueError("pooling mask contains an empty row")
    if mode in {"attention_mean", "final_user_mean"}:
        weights = valid.unsqueeze(-1).to(hidden.dtype)
        return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)
    if mode == "last_nonpadding":
        positions = torch.arange(hidden.shape[1], device=hidden.device)
        indices = (valid * positions.unsqueeze(0)).argmax(dim=1)
        return hidden[torch.arange(hidden.shape[0], device=hidden.device), indices]
    raise ValueError(f"unknown pooling mode {mode!r}")


def _tokenize(tokenizer, texts: list[str], final_user_texts: list[str], target):
    want_offsets = target.pooling == "final_user_mean"
    if want_offsets and not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("final_user_mean pooling requires a fast tokenizer")
    kwargs = {
        "return_tensors": "pt",
        "padding": True,
        "add_special_tokens": False,
        "return_offsets_mapping": want_offsets,
    }
    old_side = getattr(tokenizer, "truncation_side", "right")
    if target.overlength == "truncate_left":
        tokenizer.truncation_side = "left"
        kwargs.update(truncation=True, max_length=target.max_length)
    else:
        kwargs.update(truncation=False)
    try:
        encoded = tokenizer(texts, **kwargs)
    finally:
        tokenizer.truncation_side = old_side
    offsets = encoded.pop("offset_mapping", None)
    attention = encoded["attention_mask"].bool()
    counts = attention.sum(dim=1)
    if target.overlength == "error" and bool((counts > target.max_length).any()):
        bad = [int(v) for v in counts[counts > target.max_length].tolist()]
        raise RuntimeError(
            f"{len(bad)} target prompts exceed max_length={target.max_length}; "
            f"longest in batch is {max(bad)}. No tokens were truncated."
        )

    if target.pooling == "attention_mean" or target.pooling == "last_nonpadding":
        pool_mask = attention
    else:
        pool_mask = attention.new_zeros(attention.shape)
        for row, (rendered, final_text) in enumerate(zip(texts, final_user_texts)):
            start = rendered.rfind(final_text)
            if start < 0:
                raise RuntimeError("final user text not found in rendered chat template")
            end = start + len(final_text)
            starts = offsets[row, :, 0]
            ends = offsets[row, :, 1]
            overlap = (ends > start) & (starts < end) & (ends > starts)
            pool_mask[row] = overlap & attention[row]
            if not bool(pool_mask[row].any()):
                raise RuntimeError("tokenizer offsets found no final-user tokens")
    return encoded, pool_mask, counts, pool_mask.sum(dim=1)


class BlockCollector:
    """Pool at the exact block-output modules later used for steering."""

    def __init__(self, blocks, indices, mode):
        self.mode = mode
        self.mask = None
        self.values = {}
        self.handles = []
        for index in indices:
            self.handles.append(blocks[index].register_forward_hook(
                self._hook(index)
            ))

    def _hook(self, index):
        def collect(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            self.values[index] = pool_hidden(hidden, self.mask, self.mode).detach()
            return output
        return collect

    def begin(self, mask):
        self.mask = mask
        self.values = {}

    def stacked(self, indices):
        missing = set(indices) - set(self.values)
        if missing:
            raise RuntimeError(f"decoder hooks did not fire for blocks {sorted(missing)}")
        import torch
        return torch.stack([self.values[index] for index in indices], dim=1)

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz",
                                     dir=path.parent)
    os.close(fd)
    try:
        np.savez(temporary, **arrays)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _render_all(tokenizer, cells, table, target) -> tuple[list[str], list[str]]:
    rendered, finals = [], []
    for cell, row in zip(cells, table.itertuples(index=False)):
        semantic_digest = prompt_digest(
            deployment_messages(cell, target.answer_instruction),
            target.system_prompt,
        )
        if semantic_digest != row.semantic_prompt_digest:
            raise RuntimeError(
                f"target prompt drift for {row.cell_id}: the current deployment "
                "builder no longer matches the frozen design"
            )
        rendered.append(render_target_prompt(tokenizer, cell, target))
        finals.append(deployment_user_text(cell, target.answer_instruction))
    return rendered, finals


def _token_census(tokenizer, cells, table, target) -> tuple[pd.DataFrame, list, list]:
    rendered, finals = _render_all(tokenizer, cells, table, target)
    rows = table.copy()
    token_counts, span_counts, digests, tokenization_digests = [], [], [], []
    batch_size = max(1, target.batch_size)
    for start in range(0, len(rendered), batch_size):
        stop = min(start + batch_size, len(rendered))
        encoded, pool_mask, counts, spans = _tokenize(
            tokenizer, rendered[start:stop], finals[start:stop], target
        )
        token_counts.extend(int(v) for v in counts.tolist())
        span_counts.extend(int(v) for v in spans.tolist())
        digests.extend(hashlib.sha256(text.encode()).hexdigest()[:20]
                       for text in rendered[start:stop])
        attention = encoded["attention_mask"].bool()
        for row_index in range(len(counts)):
            active_ids = encoded["input_ids"][row_index][attention[row_index]]
            active_pool = pool_mask[row_index][attention[row_index]]
            digest = hashlib.sha256()
            digest.update(active_ids.numpy().astype(np.int32).tobytes())
            digest.update(active_pool.numpy().astype(np.uint8).tobytes())
            tokenization_digests.append(digest.hexdigest())
    rows["target_prompt_digest"] = digests
    rows["token_count"] = token_counts
    rows["pool_token_count"] = span_counts
    rows["tokenization_digest"] = tokenization_digests
    return rows, rendered, finals


def _representation_input_digest(rows: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for row in rows.itertuples(index=False):
        digest.update(
            f"{row.row_index}\0{row.cell_id}\0{row.target_prompt_digest}\0"
            f"{row.token_count}\0{row.pool_token_count}\0"
            f"{row.tokenization_digest}\n".encode()
        )
    return digest.hexdigest()


def extract_activations(config, artifacts, *, limit: int | None = None) -> dict:
    """Extract every candidate block in one forward pass per prompt batch."""
    if limit is None:
        from syco.linear_probe.labels import require_complete_labels
        require_complete_labels(config, artifacts)
    else:
        require_manifest(
            artifacts.labels.with_suffix(".manifest.json"), config, "labels"
        )
    final_manifest_path = artifacts.activations / "manifest.json"
    if final_manifest_path.is_file():
        existing_manifest = require_manifest(
            artifacts.activations / "manifest.json", config, "activations"
        )
        if not existing_manifest.get("details", {}).get("limited_debug_run"):
            return existing_manifest
    cells, table = load_frozen_cells(config, artifacts)
    total_rows = len(cells)
    if limit is not None:
        cells, table = cells[:limit], table.iloc[:limit].copy()
    limited_debug_run = len(cells) < total_rows
    tokenizer = load_tokenizer(config.target)
    rows, rendered, finals = _token_census(
        tokenizer, cells, table, config.target
    )
    representation_input_digest = _representation_input_digest(rows)

    model = load_target_model(config.target)
    blocks, block_path = resolve_decoder_blocks(model)
    indices = candidate_block_indices(len(blocks), config.target.layers)
    fingerprint = target_fingerprint(
        model, tokenizer, config.target, block_path, indices
    )
    block_count = len(blocks)
    work_manifest_path = artifacts.activations / "work_manifest.json"
    work_details = {
        "target_fingerprint": fingerprint,
        "candidate_block_indices": list(indices),
        "batch_size": int(config.target.batch_size),
        "pooling": config.target.pooling,
        "dataset_manifest_sha256": sha256_file(artifacts.dataset_manifest),
    }
    if work_manifest_path.is_file():
        work_manifest = require_manifest(
            work_manifest_path, config, "activations_work"
        )
        if work_manifest.get("details") != work_details:
            raise RuntimeError(
                "partial activation shards were produced by a different target, "
                "block set, batch size, pooling mode, or frozen dataset; use a "
                "separate output directory"
            )
    else:
        atomic_json(
            work_manifest_path,
            stage_manifest(config, "activations_work", details=work_details),
        )
    input_device = pick_input_device(model)
    collector = BlockCollector(blocks, indices, config.target.pooling)
    shard_dir = artifacts.activations / "shards"
    batch_size = max(1, config.target.batch_size)
    shard_paths, width = [], None
    try:
        import torch

        with torch.inference_mode():
            for shard_index, start in enumerate(range(0, len(cells), batch_size)):
                stop = min(start + batch_size, len(cells))
                shard_path = shard_dir / f"part-{shard_index:06d}.npz"
                expected_rows = rows.iloc[start:stop]["row_index"].to_numpy(
                    dtype=np.int64
                )
                expected_prompt_digests = rows.iloc[start:stop][
                    "target_prompt_digest"
                ].to_numpy(dtype="U20")
                expected_tokenization_digests = rows.iloc[start:stop][
                    "tokenization_digest"
                ].to_numpy(dtype="U64")
                if shard_path.is_file():
                    with np.load(shard_path) as existing:
                        old_rows = existing["row_indices"]
                        old_blocks = existing["block_indices"]
                        old_prompt_digests = existing["prompt_digests"]
                        old_tokenization_digests = existing[
                            "tokenization_digests"
                        ]
                        representations = existing["representations"]
                        exact = (
                            np.array_equal(old_rows, expected_rows)
                            and np.array_equal(old_blocks, indices)
                            and np.array_equal(
                                old_prompt_digests, expected_prompt_digests
                            )
                            and np.array_equal(
                                old_tokenization_digests,
                                expected_tokenization_digests,
                            )
                            and representations.ndim == 3
                            and representations.shape[:2]
                            == (len(expected_rows), len(indices))
                            and (
                                fingerprint["hidden_size"] is None
                                or representations.shape[-1]
                                == int(fingerprint["hidden_size"])
                            )
                            and np.isfinite(representations).all()
                        )
                        if exact:
                            shape = representations.shape
                            width = shape[-1]
                            shard_paths.append(shard_path)
                            continue
                        # A limited debug run can end in a short final batch. An
                        # unlimited resume recomputes only that extended batch.
                        extend_debug_batch = (
                            np.array_equal(old_blocks, indices)
                            and len(old_rows) < len(expected_rows)
                            and np.array_equal(old_rows, expected_rows[:len(old_rows)])
                            and np.array_equal(
                                old_prompt_digests,
                                expected_prompt_digests[:len(old_prompt_digests)],
                            )
                            and np.array_equal(
                                old_tokenization_digests,
                                expected_tokenization_digests[
                                    :len(old_tokenization_digests)
                                ],
                            )
                        )
                    if not extend_debug_batch:
                        raise ValueError(
                            f"activation shard does not match design: {shard_path}"
                        )

                encoded, pool_mask, _, _ = _tokenize(
                    tokenizer, rendered[start:stop], finals[start:stop], config.target
                )
                model_inputs = {key: value.to(input_device)
                                for key, value in encoded.items()}
                collector.begin(pool_mask)
                model(**model_inputs, use_cache=False, return_dict=True)
                stacked = collector.stacked(indices).float().cpu().numpy()
                width = stacked.shape[-1]
                _atomic_npz(
                    shard_path,
                    row_indices=expected_rows,
                    block_indices=np.asarray(indices, dtype=np.int32),
                    prompt_digests=expected_prompt_digests,
                    tokenization_digests=expected_tokenization_digests,
                    representations=stacked.astype(np.float16),
                )
                shard_paths.append(shard_path)
    finally:
        collector.close()
        del blocks
        del model
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Publish row metadata only after every requested shard has validated or
    # completed. A failed resume therefore cannot corrupt an older manifest.
    _atomic_parquet(rows, artifacts.activation_rows)
    details = {
        "rows": len(rows),
        "shards": len(shard_paths),
        "storage_dtype": "float16",
        "hidden_width": width,
        "candidate_block_indices": list(indices),
        "decoder_block_count": block_count,
        "pooling": config.target.pooling,
        "tokens": {
            "total": int(rows.token_count.sum()),
            "mean": float(rows.token_count.mean()),
            "p95": float(rows.token_count.quantile(.95)),
            "max": int(rows.token_count.max()),
        },
        "target_fingerprint": fingerprint,
        "limited_debug_run": limited_debug_run,
        "full_design_rows": total_rows,
        "representation_input_digest": representation_input_digest,
    }
    manifest = stage_manifest(
        config,
        "activations",
        inputs={
            "labels_sha256": sha256_file(artifacts.labels),
            "dataset_manifest_sha256": sha256_file(artifacts.dataset_manifest),
        },
        details=details,
    )
    manifest["shards"] = [
        {"path": str(path.relative_to(artifacts.root)), "sha256": sha256_file(path)}
        for path in shard_paths
    ]
    manifest["artifacts"] = {
        "rows_sha256": sha256_file(artifacts.activation_rows),
    }
    atomic_json(artifacts.activations / "manifest.json", manifest)
    return manifest


def load_activation_matrix(config, artifacts) -> tuple[np.ndarray, pd.DataFrame, dict]:
    manifest = require_manifest(
        artifacts.activations / "manifest.json", config, "activations"
    )
    rows = pd.read_parquet(artifacts.activation_rows).sort_values("row_index")
    if manifest.get("details", {}).get("limited_debug_run"):
        raise RuntimeError(
            "activation extraction is a limited debug run; resume extract without "
            "--limit before training"
        )
    expected_rows_hash = (manifest.get("artifacts") or {}).get("rows_sha256")
    if sha256_file(artifacts.activation_rows) != expected_rows_hash:
        raise ValueError("activation row metadata hash mismatch")
    parts = []
    row_indices = []
    for entry in manifest.get("shards", []):
        path = artifacts.root / entry["path"]
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"activation shard hash mismatch: {path}")
        with np.load(path) as shard:
            parts.append(shard["representations"].astype(np.float32))
            row_indices.append(shard["row_indices"].astype(np.int64))
    if not parts:
        raise RuntimeError("activation manifest contains no shards")
    matrix = np.concatenate(parts, axis=0)
    observed = np.concatenate(row_indices)
    expected = rows["row_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(observed, expected):
        raise ValueError("activation shards are not aligned to activation rows")
    return matrix, rows.reset_index(drop=True), manifest
