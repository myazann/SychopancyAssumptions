#!/usr/bin/env python3
"""
Train a linear probe (Ridge) on hidden-state representations to predict a score in [0,1].
"""
import os
import argparse
import json
import os
from typing import List, Tuple

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_data(path: str, score: str) -> Tuple[List[str], np.ndarray]:
    print(path)
    if not path.endswith(".csv"):
        raise ValueError("Data must be a .csv with columns: user_text and <score>")

    import pandas as pd

    df = pd.read_csv(path)
    before = len(df)
    df = df.dropna(subset=["user_text", score])
    after = len(df)
    if before > after:
        print(f"Warning: Dropped {before - after} rows with missing values")

    prompts = df["user_text"].astype(str).tolist()
    y = df[score].astype(float).to_numpy()

    if np.any(y < 0) or np.any(y > 1):
        y = np.clip(y, 0.0, 1.0)

    return prompts, y.astype(np.float32)


def pick_input_device(model) -> torch.device:
    try:
        emb = model.get_input_embeddings()
        if emb is not None and hasattr(emb, "weight") and emb.weight is not None:
            dev = emb.weight.device
            if dev.type != "meta":
                return dev
    except Exception:
        pass

    # fallback: first non-meta parameter
    for p in model.parameters():
        if p is not None and p.device.type != "meta":
            return p.device

    # final fallback
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
    """
    Returns X: [N, d_model] from hidden_states[layer_idx].
    pooling: "last" or "mean"
    """
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
        hs = out.hidden_states[layer_idx]  # [B, T, d]
        attn = enc["attention_mask"].unsqueeze(-1)  # [B, T, 1]

        if pooling == "last":
            lengths = enc["attention_mask"].sum(dim=1)  # [B]
            idx = (lengths - 1).clamp(min=0)  # [B]
            reps = hs[torch.arange(hs.size(0), device=hs.device), idx]  # [B, d]
        elif pooling == "mean":
            hs_masked = hs * attn
            reps = hs_masked.sum(dim=1) / attn.sum(dim=1).clamp(min=1)
        else:
            raise ValueError("pooling must be 'last' or 'mean'")

        X.append(reps.float().cpu().numpy())

    return np.concatenate(X, axis=0)


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
    
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    print("tp_plan in config:", getattr(config, "tp_plan", "NOT PRESENT"))
    config.tp_plan = None 
    print("tp_plan after clearing:", getattr(config, "tp_plan", "NOT PRESENT"))
    model = AutoModelForCausalLM.from_pretrained(args.model, config=config, **model_kwargs)
    model.eval()

    input_device = pick_input_device(model)
    return model, tokenizer, input_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to data.csv (needs user_text + score column)")
    ap.add_argument("--model", required=True, help="HF model name or local path")
    ap.add_argument("--out_dir", default="probe_out", help="Output directory")
    ap.add_argument("--pooling", choices=["mean", "last"], default="mean")
    ap.add_argument("--layers", default="-1,-5,-9,-13", help="Comma-separated layer indices to sweep (supports negatives)")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--alpha", type=float, default=10.0, help="Ridge regularization strength")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--score", type=str, default="validation_seeking_score")
    ap.add_argument("--trust_remote_code", action="store_true", help="Set if the model requires trust_remote_code")
    ap.add_argument("--device_map", type=str, default="auto", help="e.g. auto / balanced / sequential (accelerate)")
    ap.add_argument("--load_in_4bit", action="store_true", help="Enable 4-bit quantization on GPU (recommended for 70B)")
    ap.add_argument(
        "--attn_implementation",
        type=str,
        default=None,
        choices=[None, "sdpa", "eager", "flash_attention_2"],
        help="Optional: hint attention backend (requires support in your env).",
    )

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    prompts, y = load_data(args.data, args.score)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model, tokenizer, input_device = build_model_and_tokenizer(args)

    layer_list = [int(x) for x in args.layers.split(",") if x.strip()]

    idx = np.arange(len(prompts))
    tr_idx, te_idx = train_test_split(idx, test_size=0.2, random_state=args.seed)
    prompts_tr = [prompts[i] for i in tr_idx]
    prompts_te = [prompts[i] for i in te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    best = None
    results = []

    for layer in layer_list:
        X_tr = extract_reps(
            model, tokenizer, prompts_tr, layer, args.pooling, args.batch_size, args.max_length, input_device
        )
        X_te = extract_reps(
            model, tokenizer, prompts_te, layer, args.pooling, args.batch_size, args.max_length, input_device
        )

        probe = Ridge(alpha=args.alpha)
        probe.fit(X_tr, y_tr)
        pred = probe.predict(X_te)

        r2 = float(r2_score(y_te, pred))
        rmse = float(np.sqrt(mean_squared_error(y_te, pred)))

        w = probe.coef_.astype(np.float32) 

        rec = {"layer": layer, "r2": r2, "rmse": rmse}
        results.append(rec)
        print(f"[layer {layer}] R2={r2:.4f}  RMSE={rmse:.4f}")

        if best is None or r2 > best["r2"]:
            best = {"layer": layer, "r2": r2, "rmse": rmse, "w": w}

    if best is None:
        raise RuntimeError("No layers evaluated (check --layers).")

    # Save best direction
    direction = best["w"]
    direction = direction / (np.linalg.norm(direction) + 1e-8)  # normalize

    np.save(os.path.join(args.out_dir, "validation_direction.npy"), direction)

    meta = {
        "data": args.data,
        "model": args.model,
        "pooling": args.pooling,
        "layers_swept": layer_list,
        "ridge_alpha": args.alpha,
        "split_seed": args.seed,
        "best_layer": best["layer"],
        "best_r2": best["r2"],
        "best_rmse": best["rmse"],
        "all_results": results,
        "trust_remote_code": bool(args.trust_remote_code),
        "device_map": args.device_map,
        "load_in_4bit": bool(args.load_in_4bit),
        "attn_implementation": args.attn_implementation,
        "note": "direction is normalized probe weights; steer away by subtracting alpha * direction at best_layer",
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\nSaved:")
    print(" -", os.path.join(args.out_dir, "validation_direction.npy"))
    print(" -", os.path.join(args.out_dir, "meta.json"))
    print("Best layer:", best["layer"])


if __name__ == "__main__":
    main()

