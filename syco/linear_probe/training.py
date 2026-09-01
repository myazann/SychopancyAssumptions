"""Ridge probes with validation-only block selection and held-out evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from syco.linear_probe import dimensions_for_instruments
from syco.linear_probe.activations import load_activation_matrix
from syco.linear_probe.artifacts import (
    atomic_json,
    require_manifest,
    sha256_file,
    stage_manifest,
)


def fit_ridge_raw(X: np.ndarray, y: np.ndarray, alpha: float,
                  standardize: bool = False) -> tuple[np.ndarray, float]:
    """Fit Ridge and return coefficients in the original activation units."""
    from sklearn.linear_model import Ridge

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if standardize:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(X)
        transformed = scaler.transform(X)
        probe = Ridge(alpha=alpha, fit_intercept=True).fit(transformed, y)
        coef = np.asarray(probe.coef_, dtype=np.float64) / scaler.scale_
        intercept = float(probe.intercept_ - np.dot(
            np.asarray(probe.coef_, dtype=np.float64),
            scaler.mean_ / scaler.scale_,
        ))
    else:
        probe = Ridge(alpha=alpha, fit_intercept=True).fit(X, y)
        coef = np.asarray(probe.coef_, dtype=np.float64)
        intercept = float(probe.intercept_)
    return coef.astype(np.float32), intercept


def absolute_extreme_auc(y, prediction, low: float = .3, high: float = .7):
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    mask = np.isfinite(y) & np.isfinite(prediction) & ((y < low) | (y > high))
    classes = (y[mask] > high).astype(int)
    if mask.sum() < 2 or len(np.unique(classes)) < 2:
        return float("nan"), int((classes == 0).sum()), int((classes == 1).sum())
    return float(roc_auc_score(classes, prediction[mask])), \
        int((classes == 0).sum()), int((classes == 1).sum())


def quantile_extreme_auc(y, prediction, quantile: float = .3):
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    finite = np.isfinite(y) & np.isfinite(prediction)
    y, prediction = y[finite], prediction[finite]
    if len(y) < 2:
        return float("nan")
    low, high = np.quantile(y, [quantile, 1 - quantile])
    mask = (y <= low) | (y >= high)
    classes = (y[mask] >= high).astype(int)
    if len(np.unique(classes)) < 2:
        return float("nan")
    return float(roc_auc_score(classes, prediction[mask]))


def regression_metrics(y, prediction, *, auc_low=.3, auc_high=.7,
                       auc_quantile=.3) -> dict:
    from scipy.stats import spearmanr
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y = np.asarray(y, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    finite = np.isfinite(y) & np.isfinite(prediction)
    y, prediction = y[finite], prediction[finite]
    if not len(y):
        return {key: float("nan") for key in (
            "r2", "rmse", "mae", "spearman", "auc_absolute", "auc_quantile"
        )} | {"n": 0, "n_low": 0, "n_high": 0}
    auc, n_low, n_high = absolute_extreme_auc(y, prediction, auc_low, auc_high)
    correlation = spearmanr(y, prediction).statistic if len(y) >= 2 else float("nan")
    return {
        "n": len(y),
        "r2": float(r2_score(y, prediction)) if len(y) >= 2 else float("nan"),
        "rmse": float(math.sqrt(mean_squared_error(y, prediction))),
        "mae": float(mean_absolute_error(y, prediction)),
        "spearman": float(correlation),
        "auc_absolute": auc,
        "auc_quantile": quantile_extreme_auc(y, prediction, auc_quantile),
        "n_low": n_low,
        "n_high": n_high,
    }


def _atomic_npz(path: Path, **arrays):
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


def aggregate_labels(labels: pd.DataFrame, method: str) -> pd.DataFrame:
    valid = labels[labels.valid_completion & labels.score.notna()].copy()
    aggregation = "median" if method == "median" else "mean"
    grouped = valid.groupby(["cell_id", "dimension"], as_index=False)["score"]
    result = getattr(grouped, aggregation)()
    disagreement = valid.groupby(["cell_id", "dimension"]).agg(
        label_votes=("score", "count"),
        label_teachers=("label_teacher_id", "nunique"),
        label_std=("score", "std"),
        label_min=("score", "min"),
        label_max=("score", "max"),
    ).reset_index()
    return result.merge(disagreement, on=["cell_id", "dimension"], how="left")


def _stratum_metrics(frame: pd.DataFrame, y, prediction, training) -> list[dict]:
    records = []
    for column in ("persona_type", "prompt_type", "split"):
        if column not in frame:
            continue
        for value, indices in frame.groupby(column).groups.items():
            positions = frame.index.get_indexer(indices)
            metrics = regression_metrics(
                np.asarray(y)[positions], np.asarray(prediction)[positions],
                auc_low=training.auc_low, auc_high=training.auc_high,
                auc_quantile=training.auc_quantile,
            )
            records.append({"stratum": column, "value": str(value), **metrics})
    return records


def _best_layer(layer_results: list[dict]) -> dict:
    finite = [result for result in layer_results if np.isfinite(result["r2"])]
    if not finite:
        raise RuntimeError("validation R2 is undefined for every candidate block")
    return max(finite, key=lambda result: (result["r2"], -result["rmse"]))


def train_probes(config, artifacts) -> dict:
    from syco.linear_probe.labels import require_complete_labels

    require_complete_labels(config, artifacts)
    matrix, rows, activation_manifest = load_activation_matrix(config, artifacts)
    labels = pd.read_parquet(artifacts.labels)
    aggregated = aggregate_labels(labels, config.labeling.aggregation)
    block_indices = tuple(
        activation_manifest["details"]["candidate_block_indices"]
    )
    if matrix.shape[1] != len(block_indices):
        raise ValueError("activation layer axis does not match its manifest")
    row_position = {value: index for index, value in enumerate(rows.cell_id)}
    target_fp = activation_manifest["details"]["target_fingerprint"]
    predictions, metric_rows, probe_entries = [], [], []

    for dimension in dimensions_for_instruments(config.labeling.instruments):
        dimension_labels = aggregated[aggregated.dimension == dimension].copy()
        dimension_labels["activation_position"] = dimension_labels.cell_id.map(row_position)
        dimension_labels = dimension_labels.dropna(subset=["activation_position"])
        dimension_labels["activation_position"] = dimension_labels.activation_position.astype(int)
        frame = rows.merge(
            dimension_labels.drop(columns=["dimension"]), on="cell_id", how="inner",
            suffixes=("", "_label"),
        ).reset_index(drop=True)
        if frame.empty:
            raise RuntimeError(f"no valid labels align for {dimension}")
        positions = frame.activation_position.to_numpy(dtype=int)
        y = frame.score.to_numpy(dtype=np.float32)
        partition = frame.split.to_numpy(dtype=str)
        train_mask = partition == "train"
        validation_mask = partition == "validation"
        test_mask = partition == "test"
        if min(train_mask.sum(), validation_mask.sum(), test_mask.sum()) < 2:
            raise RuntimeError(
                f"{dimension} has too few leakage-safe rows: train={train_mask.sum()}, "
                f"validation={validation_mask.sum()}, test={test_mask.sum()}"
            )

        layer_results, selection_fits = [], {}
        shuffled_results = []
        rng = np.random.default_rng(config.training.split.seed +
                                    int(hashlib.sha256(dimension.encode()).hexdigest()[:8], 16))
        shuffled_y = rng.permutation(y[train_mask])
        for layer_position, block_index in enumerate(block_indices):
            X = matrix[positions, layer_position, :]
            coefficient, intercept = fit_ridge_raw(
                X[train_mask], y[train_mask], config.training.ridge_alpha,
                config.training.standardize,
            )
            validation_prediction = X[validation_mask] @ coefficient + intercept
            metrics = regression_metrics(
                y[validation_mask], validation_prediction,
                auc_low=config.training.auc_low,
                auc_high=config.training.auc_high,
                auc_quantile=config.training.auc_quantile,
            )
            result = {"block_index": int(block_index), **metrics}
            layer_results.append(result)
            selection_fits[block_index] = (coefficient, intercept)

            shuffled_coefficient, shuffled_intercept = fit_ridge_raw(
                X[train_mask], shuffled_y, config.training.ridge_alpha,
                config.training.standardize,
            )
            shuffled_prediction = (
                X[validation_mask] @ shuffled_coefficient + shuffled_intercept
            )
            shuffled_results.append({
                "block_index": int(block_index),
                **regression_metrics(
                    y[validation_mask], shuffled_prediction,
                    auc_low=config.training.auc_low,
                    auc_high=config.training.auc_high,
                    auc_quantile=config.training.auc_quantile,
                ),
            })

        best = _best_layer(layer_results)
        best_block = int(best["block_index"])
        layer_position = block_indices.index(best_block)
        X = matrix[positions, layer_position, :]
        selection_coefficient, selection_intercept = selection_fits[best_block]
        refit_mask = train_mask | validation_mask if \
            config.training.refit_train_validation else train_mask
        coefficient, intercept = fit_ridge_raw(
            X[refit_mask], y[refit_mask], config.training.ridge_alpha,
            config.training.standardize,
        )
        norm = float(np.linalg.norm(coefficient))
        if not np.isfinite(norm) or norm <= 1e-12:
            raise RuntimeError(f"{dimension} produced a zero/nonfinite coefficient")
        direction = coefficient / norm
        train_projection = X[refit_mask] @ direction
        projection_mean = float(train_projection.mean())
        projection_std = float(train_projection.std(ddof=1))
        test_prediction = X[test_mask] @ coefficient + intercept
        test_metrics = regression_metrics(
            y[test_mask], test_prediction,
            auc_low=config.training.auc_low,
            auc_high=config.training.auc_high,
            auc_quantile=config.training.auc_quantile,
        )
        label_std = float(y[validation_mask].std(ddof=1))
        passes_gate = bool(
            best["r2"] >= config.training.min_validation_r2_for_steering
            and label_std >= config.training.min_score_std_for_steering
            and best["n_low"] >= config.training.min_extreme_examples_per_class
            and best["n_high"] >= config.training.min_extreme_examples_per_class
        )

        probe_dir = artifacts.probes / dimension
        weights = probe_dir / "weights.npz"
        _atomic_npz(
            weights,
            coefficient=coefficient.astype(np.float32),
            intercept=np.asarray(intercept, dtype=np.float32),
            coefficient_norm=np.asarray(norm, dtype=np.float32),
            unit_direction=direction.astype(np.float32),
            selection_coefficient=selection_coefficient.astype(np.float32),
            selection_intercept=np.asarray(selection_intercept, dtype=np.float32),
            block_index=np.asarray(best_block, dtype=np.int32),
            projection_mean=np.asarray(projection_mean, dtype=np.float32),
            projection_std=np.asarray(projection_std, dtype=np.float32),
        )
        test_frame = frame.loc[test_mask].reset_index(drop=True)
        strata = _stratum_metrics(
            test_frame, y[test_mask], test_prediction, config.training
        )
        metadata = {
            "schema_version": 1,
            "config_digest": config.stage_digest("probes"),
            "pipeline_digest": config.digest,
            "dimension": dimension,
            "canonical_block_index": "zero_based_decoder_block",
            "block_index": best_block,
            "pooling": config.target.pooling,
            "ridge_alpha": config.training.ridge_alpha,
            "standardize": config.training.standardize,
            "refit_train_validation": config.training.refit_train_validation,
            "coefficient_norm": norm,
            "projection_mean": projection_mean,
            "projection_std": projection_std,
            "positive_direction_semantics": "h += strength * unit_direction moves toward higher labelled score",
            "selection_metrics": layer_results,
            "shuffled_label_selection_metrics": shuffled_results,
            "validation_best": best,
            "test_metrics": test_metrics,
            "test_strata": strata,
            "label_validation_std": label_std,
            "passes_steering_gate": passes_gate,
            "target_fingerprint": target_fp,
            "activation_manifest_sha256": sha256_file(
                artifacts.activations / "manifest.json"
            ),
            "weights_sha256": sha256_file(weights),
        }
        atomic_json(probe_dir / "metadata.json", metadata)
        probe_entries.append({
            "dimension": dimension,
            "block_index": best_block,
            "passes_steering_gate": passes_gate,
            "weights": str(weights.relative_to(artifacts.root)),
            "weights_sha256": sha256_file(weights),
            "metadata": str((probe_dir / "metadata.json").relative_to(artifacts.root)),
            "metadata_sha256": sha256_file(probe_dir / "metadata.json"),
        })
        for result in layer_results:
            metric_rows.append({"dimension": dimension, "stage": "validation",
                                **result})
        metric_rows.append({"dimension": dimension, "stage": "test",
                            "block_index": best_block, **test_metrics})
        for local_index, row in test_frame.iterrows():
            predictions.append({
                "cell_id": row.cell_id,
                "persona_type": row.persona_type,
                "persona_id": row.persona_id,
                "prompt_type": row.prompt_type,
                "prompt_id": row.prompt_id,
                "dimension": dimension,
                "block_index": best_block,
                "label": float(y[test_mask][local_index]),
                "prediction": float(test_prediction[local_index]),
                "split": "test",
            })

    artifacts.probes.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_parquet(artifacts.probes / "metrics.parquet", index=False)
    pd.DataFrame(predictions).to_parquet(
        artifacts.probes / "test_predictions.parquet", index=False
    )
    manifest = stage_manifest(
        config,
        "probes",
        inputs={
            "activations_manifest_sha256": sha256_file(
                artifacts.activations / "manifest.json"
            ),
            "labels_sha256": sha256_file(artifacts.labels),
        },
        details={
            "dimensions": len(probe_entries),
            "passed_steering_gate": sum(
                entry["passes_steering_gate"] for entry in probe_entries
            ),
            "target_fingerprint": target_fp,
        },
    )
    manifest["probes"] = probe_entries
    manifest["artifacts"] = {
        "metrics_sha256": sha256_file(artifacts.probes / "metrics.parquet"),
        "test_predictions_sha256": sha256_file(
            artifacts.probes / "test_predictions.parquet"
        ),
    }
    atomic_json(artifacts.probes / "manifest.json", manifest)
    return manifest


def load_probe(config, artifacts, dimension: str, *, allow_weak=False) -> tuple[dict, dict]:
    manifest = require_manifest(artifacts.probes / "manifest.json", config, "probes")
    entries = {entry["dimension"]: entry for entry in manifest.get("probes", [])}
    if dimension not in entries:
        raise KeyError(f"probe artifact has no dimension {dimension!r}")
    entry = entries[dimension]
    weights_path = artifacts.root / entry["weights"]
    metadata_path = artifacts.root / entry["metadata"]
    if sha256_file(weights_path) != entry["weights_sha256"]:
        raise ValueError(f"probe weights hash mismatch: {weights_path}")
    if sha256_file(metadata_path) != entry["metadata_sha256"]:
        raise ValueError(f"probe metadata hash mismatch: {metadata_path}")
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not metadata.get("passes_steering_gate") and not allow_weak:
        raise RuntimeError(
            f"{dimension} failed the preregistered steering gate; inspect "
            f"{metadata_path} or explicitly pass --allow-weak-probe"
        )
    with np.load(weights_path) as source:
        weights = {key: np.array(source[key]) for key in source.files}
    return weights, metadata
