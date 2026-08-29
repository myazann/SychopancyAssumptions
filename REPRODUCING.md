# Reproducing the structured assumptions study

This document separates three claims that are easy to conflate:

1. **Design reproducibility:** use the same people, dilemmas, factors, and
   instruments.
2. **Computational reproducibility:** use the same source, data, model,
   tokenizer, Python packages, and native inference libraries.
3. **Bitwise output reproducibility:** generate identical model tokens.

The preservation material below provides the first two for the active
structured study. The third is not claimed: acquisition used temperature 0.7
and the current runner did not give llama.cpp an explicit inference seed.

## Canonical identifiers

- Study design:
  `e7b7d58cceaba103dfede71d215198ead36a44912b9cf591735a0a3c51d93b2c`
- Acquisition source digest of the shards collected so far (schema 1, a hash
  of the whole `syco/`, `scripts/` and `config/` tree):
  `bd05a4761773f9aef398c0c2de5f7f538f7b61a10a13c3c5e1b031323ef38dc5`
- GGUF repository revision:
  `a483e9e6cbd595906af30beda3187c2663a1118c`
- GGUF SHA-256:
  `ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61`
- Tokenizer repository revision:
  `2ab40a9acc6d567889ca4d4e59feb2da56121454`

The exact tokenizer-file hashes, data hashes, package versions, native
llama.cpp library hashes, Git state, job relationships, and source inventory
are recorded in the run snapshot's `artifacts.json`,
`environment.json`, `bundle.json`, and `source-files.json`.

## Verify what is preserved

Run these commands from the repository root:

```bash
python -m syco design verify config/designs/structured-40x40.json

python -m syco snapshot verify results/snapshots/current
```

The design verifier reconstructs the entire coordinate set from the explicit
IDs and factors. It expects 40 people, 40 dilemmas, and 32,080 coordinates for
each structured instrument. The snapshot verifier hashes every captured file.

## What a run's identity covers, and what it does not

A manifest's `run_id` is a digest of what was administered: the model, the
instrument, the source data hashes, the exact coordinate set, and
`acquisition_digest` -- a hash of only the modules that decide which bytes
reach the model:

```text
syco/data.py  syco/models.py  syco/model_registry.py  syco/prompts.py
scripts/run_assumptions.py  config/models.yaml
```

Everything else in the repository is hashed separately as `repo_digest` and
recorded outside `identity`. Editing analysis code, adding a profile, or
freezing a new design therefore does not change any run's identity.

Schema 1 manifests -- every shard collected before this change, including the
live ones -- instead recorded `source_digest` over the whole tree. They are
read, compared, and continued as they are; nothing rewrites them.

Continuing an output does not require reproducing its `run_id`. The runner
keeps whatever the manifest records, appends the drift to a `revisions` list,
and instead checks the two are comparable against the data: every row already
present must be a cell the current configuration still administers, and their
stored `prompt_digest` values must still rebuild identically. A change that
would alter an observation is refused by name.

Single-writer discipline is now enforced rather than requested: `syco run`
holds an advisory `flock` on `<output>.lock` for the life of the run. Where the
filesystem does not support locking it warns and proceeds, so check the log if
you are running two jobs by hand.

One caveat remains. `slurm/run_model.sbatch` runs from the shared live
checkout, so a queued array element reads whatever is on disk when it starts,
not when it was submitted. That no longer disturbs identity or resume, but an
edit to one of the six acquisition files above, landing between two waves,
still changes what those waves administer. For a run that must be exactly
reproducible, extract a snapshot's `source.tar.gz` to a path nothing edits and
point `REPO` there.

## Reconstruct on another machine

1. Verify the run snapshot before using it.
2. Extract `source.tar.gz` from the snapshot into an empty directory. This is
   authoritative for the active study because its launch checkout was dirty;
   the recorded Git commit alone is insufficient.
3. Install Python 3.12.14 and create the frozen Python environment:

   ```bash
   uv python install 3.12.14
   CMAKE_ARGS="-DGGML_CUDA=on" uv sync \
     --frozen
   ```

   `uv.lock` freezes Python dependency resolution. CUDA driver/toolkit and
   native llama.cpp build differences can still affect generation. Compare
   the generated libraries with the hashes in `environment.json`; for a strict
   rerun, use a container or Apptainer image built once and record its digest.
4. Obtain the two source datasets and require these hashes before loading:

   ```text
   base_data_persona.gz  95e6a8fa8896c7747d8f765852c6b4fce38de776bf0b3d00d875323c70763c7e
   base_data_prompt.gz   2929b0a0e450da35dffbc6fb6dff3556a217f4916217212fcde53a160abadcb6
   ```

5. Download `unsloth/Qwen3.6-35B-A3B-GGUF` at the exact GGUF revision above,
   then require the recorded 22,134,528,992-byte artifact and SHA-256. Download
   `unsloth/Qwen3.6-35B-A3B` at the exact tokenizer revision and verify every
   tokenizer file against `artifacts.json`.
6. Verify `config/designs/structured-40x40.json`. Never resample from counts:
   the explicit IDs and coordinate digest are the study.
7. Recreate the complete 20-by-20 base shards before running the 40-by-40
   wave. The manifests and job evidence in the snapshot give the exact base and
   extension run IDs. A wave refuses to plan against a shard that is not yet
   settled, so finish each one before building the next on it.

## Current execution graph

```text
19779[4]  structured 4dims base
    └── aftercorr → 19833[4]  structured 4dims 40x40 extension

19784[4]  structured support-types base
    └── aftercorr → 19838[4]  structured support-types 40x40 extension
```

The extension shards each contain 24,040 cells. When joined to their complete
8,040-cell base, each instrument has 32,080 cells.

## Finish and seal the study

After both extension jobs complete:

```bash
python -m syco collect-extension \
  --profile structured-4dims-extension-40x40
python -m syco collect-extension \
  --profile structured-supporttypes-extension-40x40
python -m syco parse --all \
  --profile structured-4dims-extension-40x40
python -m syco parse --all \
  --profile structured-supporttypes-extension-40x40
```

Then run `python -m syco snapshot create` again with the final logs and
model/tokenizer paths. The final snapshot should additionally record SHA-256
hashes for the completed
base, extension, collected, and parsed artifacts. Keep acquisition shards
immutable after sealing them.

## Making another wave

Do not start with `--n-personas 20 --n-prompts 20` alone. First create a new
immutable design whose parent is the current design:

```bash
python -m syco design extend \
  --name structured-60x60 \
  --from config/designs/structured-40x40.json \
  --personas files/base_data_persona.gz \
  --prompts files/base_data_prompt.gz \
  --add-personas 20 \
  --add-prompts 20 \
  --seed 2000 \
  --output config/designs/structured-60x60.json
```

The command verifies the parent and data hashes, removes all already-selected
IDs from the eligible pools, and writes the full target plus coordinate digest.
The resulting lock contains:

- the parent design ID;
- all previously selected IDs;
- the newly selected disjoint IDs;
- exact data hashes and sampling seed;
- the complete target coordinate count and digest.

Only after verifying that lock should jobs be submitted. Put its path in the
profile's `design.lock` field (or pass `--design` directly); the acquisition
runner will verify it and use its exact IDs rather than sampling from counts.

List every shard already collected under `design.extend_from`, oldest first.
The runner unions their coordinates and emits only what the new target still
lacks, so waves compose without either duplicating cells or leaving holes:

```yaml
design:
  lock: config/designs/structured-60x60.json
  extend_from:
    - results/{model}/{probe}.jsonl
    - results/extensions/structured-40x40/{model}/{probe}.jsonl
```

To start the chain from a run that predates any lock, freeze what it already
covers:

```bash
python -m syco design freeze --name structured-20x20 \
  --run results/Gemma3-12B/4dims.jsonl \
  --run results/Gemma3-12B/supporttypes.jsonl \
  --output config/designs/structured-20x20.json
```
