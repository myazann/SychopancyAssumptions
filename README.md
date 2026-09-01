# SychopancyAssumptions

How do personas and prompt framing move a model's answer—and what does the
model think it is talking to when it answers? The pipeline supports both the
open-ended and structured Verbalized Assumptions instruments from Cheng et al.

The base data belongs in `files/` and is available from the
[project data folder](https://drive.google.com/drive/folders/1HpBYpVgrpgjlUB6ikdQXy2vDVTUbapBu).

## Repository layout

- [PIPELINE.md](PIPELINE.md) explains the study design, output, and advanced
  operations.
- [docs/linear_probe.md](docs/linear_probe.md) specifies the fresh-label Ridge
  probe and causal steering pipeline, its open decisions, and L40 estimate.
- `config/models.yaml` defines model aliases, backends, and VRAM estimates.
- `config/experiments/default.yaml` defines the default experiment.
- `config/designs/` contains frozen, exact study designs used directly by
  experiment profiles.
- `syco/` contains the data, model, scheduling, manifest, and parsing logic.
- `syco/sycophancy.py` scores the forced-choice answer table and joins the
  result to parsed assumptions.
- `syco/text_analysis.py` provides descriptive text features and marked-word
  associations for either persona text or model responses.
- `syco/analysis.py` holds the randomization tests behind `syco analyze`, one
  per unit of independence the design has; `syco/figures.py` and
  `syco/report.py` render its figures and its HTML report.
- `scripts/` contains the command implementations used by the CLI.
- `tests/` covers providers, parsing, resume safety, summaries, content
  analysis, and scheduling.
- `verbalizedassumptions/` is the reference implementation from Cheng et al.

## Install

The checked-in Python and dependency locks are the normal setup path. Run all
commands from the repository root:

```bash
uv python install
CMAKE_ARGS="-DGGML_CUDA=on" uv sync --frozen
uv run python -m syco doctor
```

`.python-version`, `pyproject.toml`, and `uv.lock` pin the interpreter and full
dependency graph. `requirements.txt` remains a looser pip-compatible fallback
for machines that do not use uv; it is not the environment record for a
published run.

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

The linear-probe path uses Transformers/PyTorch because GGUF inference cannot
expose block activations or accept residual hooks. Its default Llama target is
BF16; `bitsandbytes` is used only when a probe config selects 4- or 8-bit HF
loading.

## Check the experiment

These commands do not call a real model:

```bash
python -m syco models       # list model aliases and backends
python -m syco smoke        # offline generation -> parse -> summarize
python -m syco plan         # print the exact default grid
python -m syco linear-probe plan  # fresh-label probe design and workload
```

## Fresh-label linear probes

The new linear-probe pipeline starts from Qwen labels generated into its own
artifact namespace; it never imports the existing structured labels. Begin
with planning and a small strict-parser audit:

```bash
python -m syco linear-probe plan
python -m syco linear-probe label --dry-run --limit 20
python -m syco linear-probe parse-labels --dry-run --allow-partial
```

The production stages are `freeze`, `label`, `parse-labels`, `extract`,
`train`, `steer`, and `evaluate`. Do not launch them as an unreviewed chain:
the recommended first real step is a 200-call Qwen throughput/label-quality
benchmark. See [the full probe specification](docs/linear_probe.md) for exact
schemas, split rules, validation gates, artifacts, commands, and resource
estimates.

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

## Answer the three study questions at once

`topics`, `sycophancy` and `text` each answer one question and print it.
`analyze` answers all three over one shared topic space and writes them out:

```bash
python -m syco analyze \
    --model Llama-3.1-8B --model Qwen3.6-35B-A3B --model Gemma3-12B \
    --out results/analysis
```

0. what the assumptions say: the paper's word and bigram tables, per persona
   type and per side of the story, with the contrast between them;
1. how the disclosed persona type and the story's telling move the assumptions;
2. which words in a person's own transcript go with what the model assumed
   about them;
3. which assumptions travel with each model's **own** forced-choice sycophancy
   score -- a model with no collection of its own is skipped rather than
   scored against another model's.

This legacy `analyze` command deliberately leaves demographics unused: a run
draws 25 of its 200 personas, so every column becomes two or three groups of
eight. The separate fresh-label linear-probe pipeline freezes the exact selected
demographic rows for a later, larger heterogeneity study but does not use them
to select probes or steering directions.

It writes `results/analysis/KEY_FINDINGS.md`, a self-contained
`results/analysis/report.html`, a `README.md` and a set of CSVs per section,
and every figure in both a light and a dark rendering. Each section uses the
null its own unit of independence licenses -- shuffling inside person-and-dilemma
blocks, across the 25 people, or inside each dilemma. See
[PIPELINE.md](PIPELINE.md#the-three-question-analysis) for why those are three
different tests and what the two p-values on each table mean.

Sentence-transformer embeddings and the fitted topic model are cached under
`<out>/shared/cache`, so re-running with different test settings is quick;
the first run over two complete models takes roughly half an hour.

Use `--profile NAME_OR_PATH` with profile-aware commands to select another
experiment definition. Ready-to-run structured profiles are
`structured-4dims` and `structured-supporttypes`; for example:

```bash
python -m syco smoke --profile structured-4dims
python -m syco run --all --profile structured-supporttypes
python -m syco parse --all --profile structured-supporttypes
python -m syco summarize --all --profile structured-supporttypes
```

### Collect a design in waves

These runs take days, so the grid is collected in pieces. A **wave** is one
immutable acquisition shard. Each wave reads every coordinate the earlier waves
already hold, works out the full target grid, and runs only the difference --
so a second wave adding 20 people and 20 dilemmas to a 20 x 20 base does *not*
run a second isolated 20 x 20 block. It runs the missing
old-person/new-dilemma, new-person/old-dilemma, and new-person/new-dilemma
cells: 24,040 of them, giving a complete 40 x 40.

What makes that safe is a **design lock**: a small JSON file naming the exact
people, dilemmas, facets, and framings the study is aiming at, addressed by the
digest of its own coordinate set. Acquisition verifies it before planning
anything, so growing the study can never quietly resample it.

```bash
# 1. Lock the design a finished run already covers.
python -m syco design freeze --name structured-20x20 \
    --run results/Gemma3-12B/4dims.jsonl \
    --run results/Gemma3-12B/supporttypes.jsonl \
    --output config/designs/structured-20x20.json

# 2. Choose the next target: the same people and dilemmas, plus 20 more of each.
python -m syco design extend --name structured-60x60 \
    --from config/designs/structured-40x40.json \
    --personas files/base_data_persona.gz \
    --prompts files/base_data_prompt.gz \
    --add-personas 20 --add-prompts 20 --seed 2000 \
    --output config/designs/structured-60x60.json

# 3. Check what it asks for before committing any GPU time.
python -m syco design verify config/designs/structured-60x60.json
```

Then point a profile at the new lock and list every shard already collected,
oldest first:

```yaml
design:
  lock: config/designs/structured-60x60.json
  extend_from:
    - results/{model}/{probe}.jsonl                              # wave 1
    - results/extensions/structured-40x40/{model}/{probe}.jsonl  # wave 2
```

`python -m syco status --profile NAME` then reports exactly how many cells the
wave still owes, and `run` collects them. Earlier shards are opened read-only;
each wave writes only its own file.

Two rules the tooling enforces rather than documents. A wave cannot be planned
against a shard that is still being written -- that would put the same
coordinates in two files at once -- so finish a shard before building on it.
And one output has one writer: `syco run` holds an advisory lock on its JSONL
for the life of the run.

### The 40 x 40 wave, on Slurm

The two `*-extension-40x40` profiles add the exact 20 people and 20 dilemmas in
`config/designs/structured-40x40.json`, writing under
`results/extensions/structured-40x40/`. Queue them behind the base arrays:

```bash
# Defaults to the current base arrays 19779 (4dims) and 19784 (supporttypes).
# Set BASE_4DIMS_JOB / BASE_SUPPORTTYPES_JOB when those IDs change.
bash slurm/submit_structured_extension.sh
```

The script uses `aftercorr`, so a model whose base task already succeeded can
start immediately, while a model still running remains pending until its own
base array element succeeds. Once every wave is in, join them:

```bash
python -m syco collect-extension --profile structured-4dims-extension-40x40
python -m syco collect-extension --profile structured-supporttypes-extension-40x40
python -m syco parse --all --profile structured-4dims-extension-40x40
python -m syco parse --all --profile structured-supporttypes-extension-40x40
```

`collect-extension` joins every wave named in `extend_from` plus the profile's
own output, and checks the union against the lock's coordinate digest -- a real
independent check, where comparing the rows only to themselves would miss a
facet absent from every wave alike. The combined collections live under
`results/collections/structured-40x40/` with one collection run ID, while
retaining each row's `source_run_id` and `source_cell_key` for auditability.

### The 45 x 40 open-ended wave

The same mechanism applies to the open-ended probe. The `default` profile
collected 25 people x 20 dilemmas (10,040 cells per model);
`openended-extension-45x40` adds 20 more of each from
`config/designs/openended-45x40.json`, which is 26,040 further cells per model
for a complete 45 x 40.

The open-ended bases did not all finish together, so submission picks up only
the models whose base is settled and says why the others are held back:

```bash
DRY_RUN=1 bash slurm/submit_openended_extension.sh   # show the plan
bash slurm/submit_openended_extension.sh             # submit what is ready
```

Re-run it whenever a held-back base finishes; it submits only what is newly
ready, and a wave already part-collected resumes rather than restarts. The
header of that script has the two-line recipe for finishing a lagging base and
chaining its wave behind it with `--dependency=afterok`.

### Why a run keeps its identity when the code moves

Every output carries a manifest whose `run_id` is a digest of what was
administered: the model, the instrument, the source data, the exact
coordinates, and `acquisition_digest` -- a hash of just the modules that decide
which bytes reach the model. Analysis code, profiles, and the rest of the
repository are hashed separately as `repo_digest`, recorded for provenance but
kept out of the identity.

That split is what lets a study survive its own development. `cell_key` embeds
`run_id`, so recomputing a new one would orphan every row already collected.
When you re-run a command against an existing output, the runner keeps the
recorded `run_id`, appends what changed to a `revisions` list, and proves the
two are comparable against the data rather than against a hash: every prior row
must be a cell the current configuration still administers, and their stored
`prompt_digest` values must still reproduce. A change that would really alter
an observation -- a different model, probe, system prompt, or design -- is
refused, and names the field that differs.

See [REPRODUCING.md](REPRODUCING.md) for verifying the design, restoring the
locked environment, and capturing a self-verifying run snapshot under
`results/snapshots/`.
