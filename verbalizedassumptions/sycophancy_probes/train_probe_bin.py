#!/usr/bin/env python3
import os
import argparse
import json
from typing import List, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def load_data(path: str, score: str) -> Tuple[List[str], np.ndarray]:
    if not path.endswith(".csv"):
        raise ValueError("Data must be a .csv with columns: user_text and <score>")

    import pandas as pd

    df = pd.read_csv(path)
    before = len(df)
    df[score] = pd.to_numeric(df[score], errors="coerce")
    text_cols = ['user_text', 'prompt']

    nan_counts = {
            col: df[col].isna().sum() if col in df.columns else float('inf')
            for col in text_cols
        }

    text_col = min(nan_counts, key=nan_counts.get)
    df = df.dropna(subset=[text_col, score])
    after = len(df)
    if before > after:
        print(f"Warning: Dropped {before - after} rows with missing values")

    prompts = df[text_col].astype(str).tolist()
    y = df[score].astype(float).to_numpy()

    unique_vals = np.unique(y)
    if not np.all(np.isin(unique_vals, [0, 1])):
        raise ValueError(f"Score must be binary (0 or 1). Found unique values: {unique_vals}")

    return prompts, y.astype(np.int32)


def pick_input_device(model) -> torch.device:
    """Return a single device suitable for input tensors (handles device_map sharding)."""
    try:
        emb = model.get_input_embeddings()
        if emb is not None and hasattr(emb, "weight") and emb.weight is not None:
            dev = emb.weight.device
            if dev.type != "meta":
                return dev
    except Exception:
        pass

    for p in model.parameters():
        if p is not None and p.device.type != "meta":
            return p.device

    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def build_model_and_tokenizer(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_kwargs = dict(
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,
        tp_plan=None,
    )

    if device == "cuda":
        bf16_ok = torch.cuda.is_bf16_supported()
        model_kwargs["torch_dtype"] = torch.bfloat16 if bf16_ok else torch.float16

        if args.load_in_4bit:
            from transformers import BitsAndBytesConfig
            compute_dtype = torch.bfloat16 if bf16_ok else torch.float16
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
            )
            model_kwargs["quantization_config"] = bnb_config

        model_kwargs["device_map"] = args.device_map
    else:
        model_kwargs["torch_dtype"] = torch.float32
        model_kwargs["device_map"] = None

    if args.attn_implementation and device == "cuda":
        model_kwargs["attn_implementation"] = args.attn_implementation

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    print("tp_plan in config:", getattr(config, "tp_plan", "NOT PRESENT"))
    config.tp_plan = None
    print("tp_plan after clearing:", getattr(config, "tp_plan", "NOT PRESENT"))

    model = AutoModelForCausalLM.from_pretrained(args.model, config=config, **model_kwargs)
    model.eval()

    input_device = pick_input_device(model)
    return model, tokenizer, input_device


@torch.no_grad()
def extract_reps(
    model,
    tokenizer,
    prompts: List[str],
    layer_idx: int,
    pooling: str,
    batch_size: int,
    max_length: int,
    input_device: torch.device,
):
    """Returns X: [N, d_model] from hidden_states[layer_idx]."""
    X = []
    if pooling == "last":
        tokenizer.padding_side = "left"

    for i in tqdm(range(0, len(prompts), batch_size), desc=f"Extract layer={layer_idx} pool={pooling}"):
        batch = prompts[i : i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        enc = {k: v.to(input_device) for k, v in enc.items()}

        out = model(**enc, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states[layer_idx]           # [B, T, d]
        attn = enc["attention_mask"].unsqueeze(-1)  # [B, T, 1]

        if pooling == "last":
            lengths = enc["attention_mask"].sum(dim=1)
            idx = (lengths - 1).clamp(min=0)
            reps = hs[torch.arange(hs.size(0), device=hs.device), idx]
        elif pooling == "mean":
            hs_masked = hs * attn
            reps = hs_masked.sum(dim=1) / attn.sum(dim=1).clamp(min=1)
        else:
            raise ValueError("pooling must be 'last' or 'mean'")

        X.append(reps.float().cpu().numpy())

    return np.concatenate(X, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to data.csv")
    ap.add_argument("--model", required=True, help="HF model name or local path")
    ap.add_argument("--out_dir", default="probe_out", help="Output directory")
    ap.add_argument("--pooling", choices=["mean", "last"], default="mean")
    ap.add_argument("--layers", default="12,16,20,24,28,32", help="Comma-separated layer indices (supports negatives)")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--C", type=float, default=1.0, help="Inverse regularization strength for LogisticRegression")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--score", type=str, default="binary_label")
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--device_map", type=str, default="auto")
    ap.add_argument("--load_in_4bit", action="store_true", help="4-bit quantization (recommended for 70B+)")
    ap.add_argument(
        "--attn_implementation",
        type=str,
        default=None,
        choices=[None, "sdpa", "eager", "flash_attention_2"],
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    prompts, y = load_data(args.data, args.score)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model, tokenizer, input_device = build_model_and_tokenizer(args)

    layer_list = [int(x) for x in args.layers.split(",") if x.strip()]

    idx = np.arange(len(prompts))
    tr_idx, te_idx = train_test_split(idx, test_size=0.2, random_state=args.seed, stratify=y)
    prompts_tr = [prompts[i] for i in tr_idx]
    prompts_te = [prompts[i] for i in te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    print(f"Training set: {len(y_tr)} samples (class 0: {(y_tr==0).sum()}, class 1: {(y_tr==1).sum()})")
    print(f"Test set:     {len(y_te)} samples (class 0: {(y_te==0).sum()}, class 1: {(y_te==1).sum()})\n")

    best = None
    results = []

    for layer in layer_list:
        X_tr = extract_reps(model, tokenizer, prompts_tr, layer, args.pooling, args.batch_size, args.max_length, input_device)
        X_te = extract_reps(model, tokenizer, prompts_te, layer, args.pooling, args.batch_size, args.max_length, input_device)

        probe = LogisticRegression(C=args.C, max_iter=1000, class_weight="balanced", random_state=args.seed)
        probe.fit(X_tr, y_tr)

        pred = probe.predict(X_te)
        pred_proba = probe.predict_proba(X_te)[:, 1]

        accuracy  = float(accuracy_score(y_te, pred))
        precision = float(precision_score(y_te, pred, zero_division=0))
        recall    = float(recall_score(y_te, pred, zero_division=0))
        f1        = float(f1_score(y_te, pred, zero_division=0))
        auc       = float(roc_auc_score(y_te, pred_proba))

        w    = probe.coef_[0].astype(np.float32)
        bias = float(probe.intercept_[0])

        rec = {"layer": layer, "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "auc": auc}
        results.append(rec)
        print(f"[layer {layer}] Acc={accuracy:.4f}  Prec={precision:.4f}  Rec={recall:.4f}  F1={f1:.4f}  AUC={auc:.4f}")

        if best is None or f1 > best["f1"]:
            best = {"layer": layer, "accuracy": accuracy, "precision": precision,
                    "recall": recall, "f1": f1, "auc": auc, "w": w, "bias": bias}

    if best is None:
        raise RuntimeError("No layers evaluated (check --layers).")

    direction = best["w"] / (np.linalg.norm(best["w"]) + 1e-8)
    np.save(os.path.join(args.out_dir, "binary_direction.npy"), direction)
    np.save(os.path.join(args.out_dir, "binary_bias.npy"), np.array([best["bias"]], dtype=np.float32))

    meta = {
        "data": args.data,
        "model": args.model,
        "pooling": args.pooling,
        "layers_swept": layer_list,
        "logistic_C": args.C,
        "split_seed": args.seed,
        "best_layer": best["layer"],
        "best_accuracy": best["accuracy"],
        "best_precision": best["precision"],
        "best_recall": best["recall"],
        "best_f1": best["f1"],
        "best_auc": best["auc"],
        "best_bias": best["bias"],
        "all_results": results,
        "trust_remote_code": bool(args.trust_remote_code),
        "device_map": args.device_map,
        "load_in_4bit": bool(args.load_in_4bit),
        "attn_implementation": args.attn_implementation,
        "note": "direction is normalized probe weights; steer away from class 1 by subtracting alpha * direction at best_layer",
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\nSaved:")
    print(" -", os.path.join(args.out_dir, "binary_direction.npy"))
    print(" -", os.path.join(args.out_dir, "binary_bias.npy"))
    print(" -", os.path.join(args.out_dir, "meta.json"))
    print(f"Best layer: {best['layer']} (F1={best['f1']:.4f})")


if __name__ == "__main__":
    main()
