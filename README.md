# SychopancyAssumptions

How do personas and prompt framing move a model's answer—and what does the
model think it is talking to when it answers? The pipeline supports both the
open-ended and structured Verbalized Assumptions instruments from Cheng et al.

The base data belongs in `files/` and is available from the
[project data folder](https://drive.google.com/drive/folders/1HpBYpVgrpgjlUB6ikdQXy2vDVTUbapBu).

## Repository layout

- [PIPELINE.md](PIPELINE.md) explains the study design, output, and advanced
  operations.
- `config/models.yaml` defines model aliases, backends, and VRAM estimates.
- `config/experiments/default.yaml` defines the default experiment.
- `syco/` contains the data, model, scheduling, manifest, and parsing logic.
- `syco/sycophancy.py` scores the forced-choice answer table and joins the
  result to parsed assumptions.
- `syco/text_analysis.py` provides descriptive text features and marked-word
  associations for either persona text or model responses.
- `scripts/` contains the command implementations used by the CLI.
- `tests/` covers providers, parsing, resume safety, summaries, content
  analysis, and scheduling.
- `verbalizedassumptions/` is the reference implementation from Cheng et al.

## Install

Use Python 3.10 or newer and run all commands from the repository root:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m syco doctor
```

`requirements.txt` is the only requirements file: it lists everything the
pipeline uses, including the topic model, the API clients, and the test and
lint tools. It does not install the project as a package -- `python -m syco`
runs the local source directly.

`bertopic` and `sentence-transformers` are the heavy entries, because the
latter pulls in torch. They are only needed for tables 5-6 of `python -m syco
topics`; on a machine where installing torch is not worth it, drop those two
lines and that command still prints its n-gram tables and reports what it
skipped.

The standard `llama-cpp-python` installation may be CPU-only. On an NVIDIA
machine, install a CUDA-enabled wheel or build it with CUDA before running a
real local experiment. One common source-build command is:

```bash
CMAKE_ARGS="-DGGML_CUDA=on" python -m pip install --upgrade --force-reinstall \
    llama-cpp-python --no-cache-dir
```

The disabled Transformers backend is not part of the default installation. To
use an `hf` model such as `Gemma3-12B-hf`, additionally install `torch` and
`accelerate`; its configured 4-bit mode also needs `bitsandbytes`.

## Check the experiment

These commands do not call a real model:

```bash
python -m syco models       # list model aliases and backends
python -m syco smoke        # offline generation -> parse -> summarize
python -m syco plan         # print the exact default grid
```

## Run one model

This starts a resumable experiment with the smallest enabled model:

```bash
python -m syco run \
    --model Llama-3.1-8B \
    --n-personas 25 \
    --n-prompts 20
```

Choose a structured probe with the same grid and runner:

```bash
# Four sycophancy-related belief dimensions.
python -m syco run --model Llama-3.1-8B --probe 4dims \
    --n-personas 25 --n-prompts 20

# Five support-seeking dimensions.
python -m syco run --model Llama-3.1-8B --probe supporttypes \
    --n-personas 25 --n-prompts 20
```

By default this writes every model-specific artifact under
`results/Llama-3.1-8B/`; the raw file is
`openended3.jsonl`. Re-run the same command to resume. Use `--plan-only` to inspect it without
loading weights, or add `--limit 10` for a ten-cell trial.

The JSONL is the append-safe raw acquisition log; parsing creates a typed
`*_assumptions.parquet` for open-ended runs or `*_structured.parquet` for
structured runs. Keep the raw file and its manifest as the reproducible source,
and use Parquet for analysis. See
[PIPELINE.md](PIPELINE.md#why-there-are-both-jsonl-and-parquet-files) for the
schema and rationale.

## Run the configured experiment

Everything uses Python; there is no shell runner:

```bash
python -m syco run --all
python -m syco status
python -m syco merge
python -m syco parse --all
python -m syco summarize --all
python -m syco topics --all
```

Score sycophancy from the constrained Yes/No collection and rank assumptions
by that score:

```bash
python -m syco sycophancy binary files/gemma-3-12b-it_results.pkl --out b.parquet
python -m syco sycophancy join   results/Gemma3-12B/openended3_assumptions.parquet \
    --binary b.parquet
```

Analyze language separately. The same helpers accept persona transcripts or
model-response columns; none of their outputs is used as a sycophancy score:

```bash
python -m syco text features files/gemma-3-12b-it_long_results.pkl \
    --text-column model_answer --method stance --out response_features.parquet
python -m syco text words files/base_data_persona.gz \
    results/Gemma3-12B/openended3_assumptions.parquet \
    --text-column persona_text --persona-role user \
    --out persona_assumption_words.parquet
```

See [PIPELINE.md](PIPELINE.md#sycophancy-and-which-assumptions-travel-with-it)
for what each score measures and what the join can and cannot support.

Use `--profile NAME_OR_PATH` with profile-aware commands to select another
experiment definition. Ready-to-run structured profiles are
`structured-4dims` and `structured-supporttypes`; for example:

```bash
python -m syco smoke --profile structured-4dims
python -m syco run --all --profile structured-supporttypes
python -m syco parse --all --profile structured-supporttypes
python -m syco summarize --all --profile structured-supporttypes
```
