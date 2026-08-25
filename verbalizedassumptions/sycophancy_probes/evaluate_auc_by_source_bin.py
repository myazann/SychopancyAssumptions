#!/usr/bin/env python3
"""
Evaluate probe AUC by source file (binary probe, prompt-only).

"""
import os
import argparse, json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm
import matplotlib.pyplot as plt


def load_data(path: str, score_col: str = "validation_seeking_score"):
    if path.endswith(".csv"):
        import pandas as pd
        df = pd.read_csv(path)
        before = len(df)
        df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
        # Decide which text column to use based on NaN count
        text_cols = ['user_text', 'prompt']

        nan_counts = {
            col: df[col].isna().sum() if col in df.columns else float('inf')
            for col in text_cols
        }

        text_col = min(nan_counts, key=nan_counts.get)

        print(f"Using text column: {text_col}")
        print(f"NaN counts: {nan_counts}")

        # Drop rows missing text or score
        df = df.dropna(subset=[text_col, score_col])

        after = len(df)
        if before > after:
            print(f"Warning: Dropped {before - after} rows with missing values")

        # Check if source_file column exists
        if 'source_file' not in df.columns:
            raise ValueError("DataFrame must have 'source_file' column")

        prompts = df[text_col].astype(str).tolist()
        y = df[score_col].astype(float).to_numpy()
        sources = df["source_file"].astype(str).tolist()
    else:
        raise ValueError("Data must be .csv")
    return prompts, y, sources


def pick_input_device(model) -> torch.device:
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

    # Clear tp_plan from config (required for some Qwen models)
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    print("tp_plan in config:", getattr(config, "tp_plan", "NOT PRESENT"))
    config.tp_plan = None
    print("tp_plan after clearing:", getattr(config, "tp_plan", "NOT PRESENT"))

    model = AutoModelForCausalLM.from_pretrained(args.model, config=config, **model_kwargs)
    model.eval()

    input_device = pick_input_device(model)
    return model, tokenizer, input_device


@torch.no_grad()
def extract_reps(model, tokenizer, prompts, layer_idx, pooling, batch_size, max_length, input_device):
    X = []
    for i in tqdm(range(0, len(prompts), batch_size), desc=f"Extract layer={layer_idx} pool={pooling}"):
        batch = prompts[i:i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        enc = {k: v.to(input_device) for k, v in enc.items()}

        out = model(**enc, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states[layer_idx]          # [B, T, d]
        attn = enc["attention_mask"].unsqueeze(-1)

        if pooling == "mean":
            reps = (hs * attn).sum(dim=1) / attn.sum(dim=1).clamp(min=1)
        else:
            lengths = enc["attention_mask"].sum(dim=1)
            idx = (lengths - 1).clamp(min=0)
            reps = hs[torch.arange(hs.size(0), device=hs.device), idx]

        X.append(reps.float().cpu().numpy())

    return np.concatenate(X, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--probe_dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--score", type=str, default="validation_seeking_score",
                    help="Score column name")
    ap.add_argument("--quantile", type=float, default=0.3,
                    help="Top/bottom quantile for binarization (default: 0.3)")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--plot", action="store_true", help="Plot ROC curve")
    ap.add_argument("--output", type=str, default="auc_by_source.png",
                    help="Output path for plot (default: auc_by_source.png)")
    ap.add_argument("--trust_remote_code", action="store_true",
                    help="Set if the model requires trust_remote_code")
    ap.add_argument("--device_map", type=str, default="auto",
                    help="e.g. auto / balanced / sequential (accelerate)")
    ap.add_argument("--load_in_4bit", action="store_true",
                    help="Enable 4-bit quantization on GPU (recommended for 70B)")
    ap.add_argument("--attn_implementation", type=str, default=None,
                    choices=[None, "sdpa", "eager", "flash_attention_2"],
                    help="Optional: hint attention backend.")
    args = ap.parse_args()

    # ----- load meta -----
    meta_path = os.path.join(args.probe_dir, "meta.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)

    layer = meta["best_layer"]
    pooling = meta["pooling"]

    # ----- load data -----
    prompts, y, sources = load_data(args.data, args.score)

    # ----- load model -----
    model, tokenizer, input_device = build_model_and_tokenizer(args)

    # ----- extract all reps -----
    print("Extracting representations...")
    X = extract_reps(
        model,
        tokenizer,
        prompts,
        layer_idx=layer,
        pooling=pooling,
        batch_size=args.batch_size,
        max_length=args.max_length,
        input_device=input_device,
    )

    # ----- probe scores = logistic regression prediction -----
    direction = np.load(os.path.join(args.probe_dir, "binary_direction.npy"))
    bias = np.load(os.path.join(args.probe_dir, "binary_bias.npy"))[0]

    # Compute logits
    logits = X @ direction + bias

    # Convert to probabilities (sigmoid)
    scores = 1 / (1 + np.exp(-logits))

    # ----- get unique sources -----
    unique_sources = sorted(set(sources))

    print(f"\nFound {len(unique_sources)} unique source files")
    print("=" * 60)

    overall_results = []

    for source in unique_sources:
        # Filter data for this source
        source_mask = np.array([s == source for s in sources])
        y_source = y[source_mask]
        scores_source = scores[source_mask]

        # Binarize labels
        lo = np.quantile(y_source, args.quantile)
        hi = np.quantile(y_source, 1 - args.quantile)

        bin_mask = (y_source <= lo) | (y_source >= hi)

        if bin_mask.sum() < 2:
            print(f"\n{source}: Insufficient data (only {bin_mask.sum()} examples)")
            continue

        y_bin = (y_source[bin_mask] >= hi).astype(int)
        scores_bin = scores_source[bin_mask]

        # Skip if only one class
        if len(np.unique(y_bin)) < 2:
            print(f"\n{source}: Only one class present after binarization")
            continue

        auc = roc_auc_score(y_bin, scores_bin)

        print(f"\n{source}:")
        print(f"  Total examples: {source_mask.sum()}")
        print(f"  Used for AUC: {bin_mask.sum()} "
              f"(bottom {args.quantile:.0%} vs top {args.quantile:.0%})")
        print(f"  ROC-AUC: {auc:.4f}")

        overall_results.append({
            'source': source,
            'auc': auc,
            'n_total': source_mask.sum(),
            'n_used': bin_mask.sum(),
            'y_bin': y_bin,
            'scores_bin': scores_bin
        })

    # ----- overall AUC -----
    print("\n" + "=" * 60)
    print("OVERALL (all sources combined):")

    lo_overall = np.quantile(y, args.quantile)
    hi_overall = np.quantile(y, 1 - args.quantile)
    mask_overall = (y <= lo_overall) | (y >= hi_overall)
    y_bin_overall = (y[mask_overall] >= hi_overall).astype(int)
    scores_overall = scores[mask_overall]

    auc_overall = roc_auc_score(y_bin_overall, scores_overall)
    print(f"  Total examples: {len(y)}")
    print(f"  Used for AUC: {mask_overall.sum()}")
    print(f"  ROC-AUC: {auc_overall:.4f}")

    # ----- plotting -----
    if args.plot and overall_results:
        plt.figure(figsize=(10, 8))

        for result in overall_results:
            fpr, tpr, _ = roc_curve(result['y_bin'], result['scores_bin'])
            plt.plot(fpr, tpr, label=f"{result['source']}: AUC={result['auc']:.3f}")

        # Add overall
        fpr, tpr, _ = roc_curve(y_bin_overall, scores_overall)
        plt.plot(fpr, tpr, 'k--', linewidth=2,
                 label=f"Overall: AUC={auc_overall:.3f}")

        plt.plot([0, 1], [0, 1], ":", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Validation-seeking probe ROC by source")
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(args.output, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {args.output}")
        plt.show()

    # ----- print summary table -----
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    print(f"{'Source':<40} {'AUC':>8} {'N_total':>10} {'N_used':>10}")
    print("-" * 60)
    for result in overall_results:
        print(f"{result['source']:<40} {result['auc']:>8.4f} {result['n_total']:>10} {result['n_used']:>10}")
    print("-" * 60)
    print(f"{'OVERALL':<40} {auc_overall:>8.4f} {len(y):>10} {mask_overall.sum():>10}")
    print("=" * 60)


if __name__ == "__main__":
    main()
