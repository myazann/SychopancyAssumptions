#!/usr/bin/env python3
import os
import argparse, json, random
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from tqdm import tqdm


def load_prompts(path: str, prompt_col: str):
    if path.endswith(".csv"):
        import pandas as pd
        df = pd.read_csv(path)
        if prompt_col not in df.columns:
            raise ValueError(f"CSV missing column '{prompt_col}'. Columns: {list(df.columns)}")
        prompts = df[prompt_col].astype(str).tolist()
        ids = df.index.astype(int).tolist()
        return ids, prompts

    if path.endswith(".jsonl"):
        ids, prompts = [], []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                obj = json.loads(line)
                if prompt_col not in obj:
                    raise ValueError(f"JSONL line {i} missing key '{prompt_col}'. Keys: {list(obj.keys())}")
                ids.append(obj.get("id", i))
                prompts.append(str(obj[prompt_col]))
        return ids, prompts

    raise ValueError("data must be .csv or .jsonl")


def parse_alphas(alpha_str: str):
    return [float(x.strip()) for x in alpha_str.split(",") if x.strip()]


def build_chat(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": "You are a helpful assistant. Reply to the user."},
         {"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def iter_batches(items, batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


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
    tokenizer.padding_side = "left"
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

        if args.use_4bit:
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

    if args.attn_impl:
        model_kwargs["attn_implementation"] = args.attn_impl

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    print("tp_plan in config:", getattr(config, "tp_plan", "NOT PRESENT"))
    config.tp_plan = None
    print("tp_plan after clearing:", getattr(config, "tp_plan", "NOT PRESENT"))

    model = AutoModelForCausalLM.from_pretrained(args.model, config=config, **model_kwargs)
    model.eval()

    input_device = pick_input_device(model)
    return model, tokenizer, input_device


def register_single_alpha_hook(model, layer_idx: int, direction: torch.Tensor):
    state = {"alpha": 0.0}
    direction = direction / (direction.norm() + 1e-8)

    def hook_fn(module, inp, out):
        a = state["alpha"]
        if a == 0.0:
            return out
        if isinstance(out, tuple):
            h = out[0]
            h = h - a * direction.to(device=h.device, dtype=h.dtype)
            return (h, *out[1:])
        else:
            return out - a * direction.to(device=out.device, dtype=out.dtype)

    handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)
    return handle, state


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to original data (.csv or .jsonl)")
    ap.add_argument("--prompt_col", default="user_text", help="Column/key name for prompts")
    ap.add_argument("--n", type=int, default=0, help="How many prompts to sample (0 = all)")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--model", required=True, help="HF model name/path")
    ap.add_argument("--probe_dir", required=True, help="Directory containing binary_direction.npy (+ meta.json)")
    ap.add_argument("--layer", type=int, default=-1, help="Override layer; default uses meta.json best_layer")

    ap.add_argument("--alphas", default="0,0.5,1,2,4,8", help="Comma-separated alpha values (include 0)")
    ap.add_argument("--out_jsonl", default="alpha_generations.jsonl")
    ap.add_argument("--out_csv_wide", default="", help="Optional: also write a wide CSV (one row per prompt)")

    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--do_sample", action="store_true", help="Use sampling (recommended); else greedy")

    ap.add_argument("--batch_size", type=int, default=16, help="Batch size for generation")
    ap.add_argument("--use_4bit", action="store_true", help="Load in 4-bit (recommended for 70B+)")
    ap.add_argument("--device_map", type=str, default="auto")
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--attn_impl", default="",
                    help="Attention backend, e.g. 'flash_attention_2' (if supported)")
    args = ap.parse_args()

    # perf knobs
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # ----- load prompts -----
    ids, prompts = load_prompts(args.data, args.prompt_col)
    if len(prompts) == 0:
        raise ValueError("No prompts loaded.")

    rng = random.Random(args.seed)
    idxs = list(range(len(prompts)))
    rng.shuffle(idxs)
    if args.n > 0:
        idxs = idxs[:args.n]
    sample = [(ids[i], prompts[i]) for i in idxs]

    # ----- load model -----
    model, tokenizer, input_device = build_model_and_tokenizer(args)

    # ----- load binary direction + choose layer -----
    direction_np = np.load(os.path.join(args.probe_dir, "binary_direction.npy")).astype(np.float32)
    direction = torch.tensor(direction_np, device=input_device, dtype=torch.float32)

    bias_np = np.load(os.path.join(args.probe_dir, "binary_bias.npy")).astype(np.float32)
    bias = float(bias_np[0])

    best_layer = None
    meta_path = os.path.join(args.probe_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        best_layer = int(meta.get("best_layer", -1))


    layer = args.layer if args.layer >= 0 else (best_layer if best_layer is not None and best_layer >= 0 else 20)
    n_layers = len(model.model.layers)
    layer = min(layer, n_layers - 2)
    # ----- generation kwargs -----
    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=bool(args.do_sample),
        temperature=args.temperature,
        top_p=args.top_p,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )

    alphas = parse_alphas(args.alphas)
    if 0.0 not in alphas:
        alphas = [0.0] + alphas

    # ----- outputs -----
    os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)
    want_wide = bool(args.out_csv_wide)
    wide = {} if want_wide else None

    # ----- single hook, alpha as outer loop -----
    hook_handle, hook_state = register_single_alpha_hook(model, layer, direction)

    try:
        sample_chats = [(pid, prompt, build_chat(tokenizer, prompt)) for pid, prompt in sample]

        with open(args.out_jsonl, "w", encoding="utf-8") as f:
            for a in tqdm(alphas, desc="Alphas"):
                hook_state["alpha"] = float(a)

                for batch in iter_batches(sample_chats, args.batch_size):
                    pids    = [x[0] for x in batch]
                    prompts_b = [x[1] for x in batch]
                    chats_b = [x[2] for x in batch]

                    inputs = tokenizer(
                        chats_b,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                    ).to(input_device)

                    gen = model.generate(**inputs, **gen_kwargs)

                    in_lens = inputs["attention_mask"].sum(dim=1).tolist()
                    for i, (pid, prompt, L) in enumerate(zip(pids, prompts_b, in_lens)):
                        new_tokens = gen[i, L:]
                        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

                        rec = {
                            "id": pid,
                            "alpha": float(a),
                            "layer": int(layer),
                            "bias": bias,
                            "prompt": prompt,
                            "output": text,
                        }
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                        if want_wide:
                            row = wide.setdefault(pid, {"id": pid, "prompt": prompt})
                            row[f"alpha_{float(a)}"] = text

        print(f"Wrote JSONL: {args.out_jsonl}")

        if want_wide:
            import pandas as pd
            df_wide = pd.DataFrame(list(wide.values()))
            df_wide.to_csv(args.out_csv_wide, index=False)
            print(f"Wrote wide CSV: {args.out_csv_wide}")

        print(f"Used layer={layer} | bias={bias:.4f} | alphas={alphas} | batch_size={args.batch_size} | "
              f"do_sample={args.do_sample} | use_4bit={args.use_4bit} | "
              f"device_map={args.device_map} | trust_remote_code={args.trust_remote_code} | "
              f"attn_impl={args.attn_impl or 'default'}")
    finally:
        hook_handle.remove()


if __name__ == "__main__":
    main()
