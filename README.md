# SychopancyAssumptions

How do personas and prompt framing move a model's answer — and what does the
model think it is talking to when it answers?

The base data (`files/`) is here:
https://drive.google.com/drive/folders/1HpBYpVgrpgjlUB6ikdQXy2vDVTUbapBu

- **[PIPELINE.md](PIPELINE.md)** — the open-ended verbalized-assumptions probe:
  design, how to run it, what comes out.
- `syco/` — the study package: data loading, the probe, the design grid, the
  model layer (hf / llama.cpp / OpenAI / Anthropic), results store, parser.
- `scripts/` — `run_assumptions.py`, `parse_assumptions.py`,
  `summarize_assumptions.py`.
- `verbalizedassumptions/` — the reference implementation from Chen et al.,
  *Verbalizing LLMs' assumptions to explain and control sycophancy*.
