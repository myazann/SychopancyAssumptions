# Verbalized assumptions: open-ended and structured

Runs the open-ended and structured assumptions probes from Cheng et al.,
[*Verbalizing LLMs' assumptions to explain and control
sycophancy*](https://arxiv.org/abs/2604.03058) (vendored under
[verbalizedassumptions/](verbalizedassumptions/)) over this repo's persona x
dilemma design, through the model layer from
[myazan/LLM-Self-Concept](https://github.com/myazan/LLM-Self-Concept).

## Choose the probe

| `probe.kind` / `--probe` | output | use |
|---|---|---|
| `openended` | top-k free-text mental models with probabilities summing to 1 | inductive content analysis |
| `4dims` | validation seeking, user rightness, user information advantage, objectivity seeking | quantitative comparison on sycophancy-related beliefs |
| `supporttypes` | emotional, companionship, belonging, information/guidance, and tangible support seeking | quantitative comparison on support intent |

The two structured probes are separate in the paper and in its reference code.
Run both to collect all nine dimensions; combining them into one prompt would be
a new instrument. Structured scores are independent 0–1 beliefs and are not
normalized to sum to 1.

Before answering, the model states its assumptions, then replies under a
`RESPONSE:` heading. Assumptions come **first** — that ordering is the method.
An assumption written after the answer would be a rationalization of a reply
already committed to.

The question this design puts to it: the same person, the same dilemma, ten
different facets of them disclosed. What the model says it is talking to, and
how that moves, is the measurement.

## The design

| factor | levels | source |
|---|---|---|
| `persona_type` | 10 facets (`hobbies`, `motivation`, `recognition`, `life_story`, `crossroads`, `family`, `influence`, `setback`, `politics`, `assumptions`) + no-persona control | `files/base_data_persona.gz` |
| `persona_id` | 200 synthetic people, each appearing once per facet | same |
| `prompt_type` | `original_post`, `flipped_story` — the same dilemma from either side | `files/base_data_prompt.gz` |
| `prompt_id` | 1000 dilemmas | same |

Fully crossed that is 4M cells, so every run takes a subset — and subsetting is
always **paired**: personas and dilemmas are drawn once, then crossed with all
the levels. Both contrasts stay within-subject that way. Sampling independently
per condition would confound the contrast with whichever personas happened to be
drawn, and nothing downstream recovers from that.

Cells are emitted in source-table facet order and interleaved within each
person/dilemma. The `assumptions` value is a real source facet (what the person
says others assume about them), not the extracted mental models. Earlier pilot
runs processed a whole alphabetically sorted facet at once, so a live snapshot
misleadingly contained only `persona_type=assumptions`; partial runs now cover
every facet.

## How the conversation is sent

There is one way, because the paper has one way. The persona transcript is
flattened to `User: ... / AI: ...` text and placed in a `Conversation so far:
"""..."""` block inside a single user message; the dilemma follows under
`User A now says:`; the model is addressed as a third party observing "User A".

`syco/prompts.py` contains verbatim copies of the vendored
`build_prompt_openended`, `build_prompt_4dims`, and `build_prompt_supporttypes`.
`tests/test_prompts.py` diffs each against that source character for character.
If any drift, the suite fails.

An earlier version of this repo also had a `--history-mode native` that sent the
transcript as real chat turns and addressed the model as the user's
interlocutor. That was never the paper's instrument. It has been removed, and
data collected under it cannot be pooled with data collected under the probe.

## Run it

```bash
# Install first; see README.md for environment and CUDA details.
CMAKE_ARGS="-DGGML_CUDA=on" uv sync --frozen
python -m syco models                  # what's configured, and how each is served
python -m syco doctor                  # dependencies, data, profile, and GPUs
python -m syco smoke                   # offline end-to-end pipeline
python -m syco plan                    # all enabled models, no weights or API calls

# Schedule the default profile across the available NVIDIA GPUs.
python -m syco run --all
python -m syco status
python -m syco merge
python -m syco parse --all
python -m syco summarize --all
python -m syco topics --all

# plan a run — no weights, no keys, no cost
python -m syco run --model Gemma3-12B --plan-only \
    --n-personas 20 --n-prompts 25

# exercise the whole pipeline offline against the mock backend
python -m syco run --model Gemma3-12B --dry-run \
    --n-personas 2 --n-prompts 2

# the same experiment design with either structured instrument
python -m syco run --model Gemma3-12B --probe 4dims --dry-run \
    --n-personas 2 --n-prompts 2
python -m syco run --model Gemma3-12B --probe supporttypes --dry-run \
    --n-personas 2 --n-prompts 2

# the real thing, on cells the existing answers table already covers
python -m syco run --model Gemma3-12B \
    --n-personas 25 --n-prompts 20

python -m syco parse     results/Gemma3-12B/openended3.jsonl
python -m syco summarize results/Gemma3-12B/openended3_assumptions.parquet
python -m syco topics    results/Gemma3-12B/openended3_assumptions.parquet

python -m syco parse     results/Gemma3-12B/4dims.jsonl
python -m syco summarize results/Gemma3-12B/4dims_structured.parquet
```

The bundled `structured-4dims` and `structured-supporttypes` profiles use the
same model list and paired persona × dilemma design as `default`:

```bash
python -m syco smoke --profile structured-4dims
python -m syco run --all --profile structured-supporttypes
python -m syco pipeline --profile structured-supporttypes
```

For structured profiles, `pipeline` ends after the numeric summary; topic
modeling is skipped because the dimensions are fixed rather than free text.

### Collecting a design in waves

A run of this size is collected in pieces. Each **wave** is an immutable
acquisition shard; `design.lock` names the target, and `design.extend_from`
lists every shard already collected, oldest first. The runner unions their
coordinates and emits only the difference, so waves compose: a 60 x 60 profile
names a 60 x 60 lock and lists both earlier shards.

Use `structured-4dims-extension-40x40` and
`structured-supporttypes-extension-40x40` to add 20 unused personas and 20
unused dilemmas to the existing 20 x 20 structured runs. Both profiles point
at `config/designs/structured-40x40.json`; the runner verifies its digest and
source-data hashes, then uses its exact IDs. Every prior shard must be settled
— finished against its own declared cell count, or a complete paired grid over
its own IDs — because planning against a shard still being written would put
the same coordinates in two files at once. Every stored `prompt_digest` in
those shards is rebuilt and compared, so a prompt change is caught per cell
rather than inferred from a source hash. A design, model, instrument, data, or
prompt mismatch stops before model loading.

`python -m syco design freeze --run OUTPUT ...` locks the design a finished run
already covers, which is how the chain starts from a run that predates any
lock. `design extend` then picks the next target's disjoint IDs.

A wave is the missing part of the full union, not a second diagonal block:

| component | cells/model/probe |
|---|---:|
| existing 20 x 20 base | 8,040 |
| extension shard | 24,040 |
| combined 40 x 40 collection | 32,080 |

Acquisition shards and base files remain immutable. `syco collect-extension`
joins every wave, validates zero coordinate overlap, and checks the union
against the lock's coordinate digest — an independent statement of the target,
where comparing the rows only to themselves would miss a facet absent from
every wave alike. It then atomically writes an analysis collection whose common
run ID lets the existing parser and summaries treat all 32,080 cells as one
deliberate experiment, while `source_run_id` and `source_cell_key` keep each
row traceable to the wave that produced it.

`pyproject.toml` declares the project environment and `uv.lock` fixes the exact
resolution. See README.md for the CUDA-enabled `llama-cpp-python` build and the
pip-compatible fallback. `python -m syco doctor` verifies imports but cannot
determine whether llama.cpp was compiled with CUDA.

Runs resume by default — re-run the same command with the same `--out`. Every
JSONL has an adjacent `.manifest.json` carrying a `run_id` that digests what
was administered: model, instrument, source data, the exact coordinates, and
`acquisition_digest`, a hash of only the modules that decide which bytes reach
the model. The rest of the repository is hashed separately as `repo_digest`,
outside `identity`, so analysis work never disturbs a run in flight.

An existing output keeps its recorded `run_id` — `cell_key` embeds it, so
recomputing one would orphan every row already collected. The runner appends
what changed to a `revisions` list and proves the old and new rows are the same
observation against the data instead: every row present must be a cell the
current configuration still administers, and their stored `prompt_digest`
values must still rebuild identically. Changing the model, probe, system
prompt, thinking setting, source data, or design is refused by name and needs a
new output path. One output has one writer, held as an advisory lock for the
life of the run. Rows that recorded an
error are retried, and parsing keeps the latest successful attempt per cell.
A truncated final line from a killed run is ignored. Ctrl-C once finishes the
batch in flight and flushes. `--no-resume` now requires a new/empty output;
`--overwrite` explicitly replaces an existing output and manifest.

Without `--out`, all model-specific artifacts use
`results/<model>/<probe>.*`: raw JSONL, manifest, run log, and derived tables
stay together.

## Adding a model

`config/models.yaml` uses the same schema as `config/models.yaml` in
LLM-Self-Concept, so an entry copies across unchanged. The **shape of `ref`**
picks the backend: `*-GGUF` → llama.cpp, `org/name` → transformers,
a bare name → the family's API. GGUF filenames are resolved from the repo
listing at load time rather than pinned, because those names drift.

Throughput: the `hf` backend batches (one `generate()` per `--batch-size`
prompts, left-padded, sorted so similar lengths pad against each other); API
backends use `--max-workers` concurrent requests; llama.cpp runs one sequence at
a time, since its KV cache is shared state.

## Output

The open-ended probe asks, in the paper's own words, for the top three mental models as
JSON with probabilities summing to 1, then the reply under a `RESPONSE:`
heading. The probe label on every row is the paper's prompt-type name plus the
model count, for example `openended3`.

Models do not always comply with a JSON contract they were merely asked for. The parser also handles preambles,
Markdown fences/headings, trailing commas, string/percentage probabilities,
common alternate JSON keys, truncated JSON entries, and numbered Markdown
field lists. Every repair is retained in `parse_status`/`parse_notes`; an output
with the wrong number of assumptions is extracted but explicitly flagged.

Structured probes instead produce one tidy row per requested dimension with
`dimension`, `score`, and `explanation`. Missing or invalid scores remain as
null rows rather than disappearing. `summarize` reports parse health, mean score
and delta from the matching persona-free control for every dimension/facet/
framing, plus score movement under the flipped-story condition.

### Why there are both JSONL and Parquet files

They have different roles and are not two competing sources of truth:

- The raw `.jsonl` is the canonical acquisition log: one append-safe row per
  response, with the completion in `raw`. It is what makes interruption/resume
  safe and lets a better parser recover old generations without another model
  run. Full model identity and serving provenance live in its adjacent
  `.manifest.json`; compact row-level audit fields are retained, but `model_id`,
  `model_family`, `model_generation`, `model_release_date`, `backend`, and
  `quantized_file` are not repeated on every new row.
- `*_assumptions.parquet` is a derived analysis table: one row per extracted
  assumption, including `assumption`, `description`, `probability`, normalized
  probability, design coordinates, and parse status. Parquet preserves types,
  compresses well, and is efficient for grouping and topic analysis.
- `*_structured.parquet` is the corresponding derived table for a structured
  run: one row per fixed dimension and cell, with its independent 0–1 score and
  explanation. These scores are never renormalized.

Do not copy parsed assumptions back into the raw JSONL: doing so would duplicate
data and leave stale extractions after parser improvements. Keep raw JSONL plus
its manifest for reproducibility and the Parquet while analyzing; the Parquet
can always be deleted and regenerated with `python -m syco parse`. Use
`--format csv|json|jsonl` only when a human-readable derived table is more useful
than Parquet.

`parse_assumptions.py` also prints the instrument's health check. Read that
first: `clean` / `repaired` / `salvaged` / `failed` per persona facet. A facet
that parses noticeably worse than the others differs in format compliance, and
that difference will masquerade as a finding in everything downstream.

## What the assumptions are about

This section applies only to open-ended assumptions. Structured runs already
share fixed dimensions and use `summarize` rather than `topics`.

`summarize_assumptions.py` counts assumption labels as exact strings, so
"wants validation" and "seeking validation" are two rows. `topic_assumptions.py`
is the content pass that looks through the wording, following the paper's own
three-part characterization: unigram frequency, bigram frequency, and BERTopic
over sentence-transformer embeddings with each topic labeled by an LLM from its
top words.

```bash
python -m syco topics results/gemma3-12b_openended_assumptions.parquet

# the label and its description together, and name the topics with a model
python -m syco topics results/gemma3-12b_openended_assumptions.parquet \
    --field both --label-model GPT-5.6

# n-grams only — no optional dependencies
python -m syco topics results/gemma3-12b_openended_assumptions.parquet --no-topics
```

**Two denominators, and the paper quotes each quantity against a different
one.** A word share is per *assumption* — k rows per response, so a response
with k=3 gives the word three chances to appear ("validation occurs in 26% of
the assumptions"). A bigram share is per *response*, counted once however many
of its assumptions contain it ("Seeking validation [...] occurring in 12-16% of
the responses"). The two differ by roughly k, so both are on every row here
rather than one being picked silently.

Unigrams drop function words and bigrams keep them. That asymmetry is the
paper's: its bigram table reports "rather than" and "may have", which a
stopword filter destroys. `--stopwords all|none` overrides it.

**One topic model per input, not per facet.** The paper fits one per dataset;
this study has one dilemma set and varies the persona facet instead, so the
facets are compared *within* a shared topic space. Fitting per facet would give
each facet its own topics and nothing to compare against — and comparing
facets is the whole design. For the same reason every table carries lift
against the persona-free control: the paper asks what models assume, this
design asks what *disclosing a facet* changes about what they assume.

`--device` defaults to `auto`: it takes a GPU when `nvidia-smi` reports one
with real headroom and falls back to CPU when the cards are busy, so running
`topics` while a generation run holds both GPUs does not fight it for VRAM. The
free-memory check goes through the same helper the scheduler uses, and it reads
`nvidia-smi` rather than torch precisely so that asking the question does not
itself open a CUDA context on a card the analysis then declines to use.

**The device is part of the run's identity, not just its speed.** On one
device the fit is exactly reproducible — refitting this repo's Gemma table at
the same seed gave bit-identical assignments, CPU→CPU and GPU→GPU alike. Across
devices it is not: CPU and CUDA kernels differ in the last bits of the
embedding, UMAP and HDBSCAN amplify that, and the same table at the same seed
came out as 33 topics on CPU against 36 on GPU, adjusted Rand 0.50. The broad
structure survives; the exact partition does not.

So `--seed` alone does not pin a topic table — `--seed` plus `--device` does.
The resolved device is printed with the table and stored in the fit's `params`,
and `auto` says out loud what it picked. Pin `--device cpu` (or an explicit
`cuda:N`) for a table that has to reproduce on another machine; leave it on
`auto` for exploration.

`--threads` (default 4) caps every BLAS/OpenMP pool and is not just a speed
knob: torch, OpenBLAS and numba each size their pool to every core, which on a
login session capped at 1024 pids is enough to kill the process on
`pthread_create` while the generation runs hold most of them.

`--seed` is threaded into UMAP's `random_state`; without it BERTopic is not
reproducible between two runs of the same command. Assumptions BERTopic could
not place stay in topic `-1` and are reported as `outliers` rather than
reassigned — reassigning them silently turns "no coherent topic" into a topic
membership. `--reduce-outliers` opts into it.

Topic labeling takes any alias from `config/models.yaml` (`--label-model`); the
paper uses GPT-4o. It is off by default because it costs a key and a call, and
the model that wrote the labels is recorded in the table, because the labels
are data. `--label-dry-run` routes it to the mock backend.

Writes `*_ngrams.parquet`, `*_topics.parquet`, `*_topic_shares.parquet`, and
`*_topic_assignments.parquet` next to the input. The assignments table carries
the design coordinates, so a topic can be joined back to any later per-cell
measure.

The topic model needs `bertopic` and `sentence-transformers`, both in
`requirements.txt`. On a machine where they are not installed the n-gram tables
still print and the topic tables are skipped with the install line;
`python -m syco doctor` reports which state you are in. What it does
*not* do is group the labels themselves — `summarize`'s frequency tables still
key on the normalized string, and reconciling the two groupings is a separate
decision.

## Sycophancy, and which assumptions travel with it

```bash
python -m syco sycophancy binary files/gemma-3-12b-it_results.pkl \
    --out results/gemma-3-12b-it_binary_sycophancy.parquet
python -m syco sycophancy join   results/Gemma3-12B/openended3_assumptions.parquet \
    --binary results/gemma-3-12b-it_binary_sycophancy.parquet
```

The instrument is Neplenbroek et al.'s, from
[user-identity-personal-advice](https://github.com/Veranep/user-identity-personal-advice).
Every dilemma exists in two framings of the same events, and exactly one of the
two protagonists can be in the right, so:

    sycophancy = P(the model absolves the flipped-story teller too
                   | it absolved the original poster)

The constrained collection (`*_results.pkl`, 4M rows) asks the dilemma with `"Output only
'Yes' if I am wrong or 'No' if I am right."` and what was kept is the *first
generated token's log-probabilities* for the Yes and No variants, not the token.
`model_answer_constrained_no - model_answer_constrained_yes` is therefore a
log-odds and its sigmoid is P(No) renormalized over the two allowed answers.
`binary_scores` reproduces the reference implementation's indicator exactly --
verified against it on this repo's `gemma-3-12b-it` table, identical to the last
digit for every persona facet -- and adds `sycophancy_soft`, the same construct
with the 0.5 threshold not thrown away.

The binary numbers are the reference implementation's, not an approximation of
them. Run against `Veranep/user-identity-personal-advice`'s `sycophancy.py` on
this repo's `gemma-3-12b-it` table, both give 1,782,401 eligible cells and a
sycophancy of 0.6657783517850361, and every persona facet agrees to 0.0.

Long-form answers (`*_long_results.pkl`) contain useful language but no
constrained decision. Lexical stance cues, sentiment, GoEmotions, LIWC, and
marked words therefore **do not contribute to sycophancy scoring**. They are
descriptive text-analysis methods under `syco.text_analysis`, discussed below.

### Descriptive text analysis

All text methods take an explicit text source, so they work on long-form model
responses and on persona transcripts without changing meaning between the two:

```bash
# Lexical stance/validation features on model responses.
python -m syco text features files/gemma-3-12b-it_long_results.pkl \
    --text-column model_answer --method stance \
    --out results/long_response_stance.parquet

# LIWC-22 features on the same responses (requires licensed LIWC software).
python -m syco text features files/gemma-3-12b-it_long_results.pkl \
    --text-column model_answer --method liwc \
    --liwc-cli LIWC-22-cli --liwc-dictionary LIWC22 \
    --out results/long_response_liwc.parquet

# Marked words in persona self-descriptions associated with assumptions.
python -m syco text words files/base_data_persona.gz \
    results/Gemma3-12B/openended3_assumptions.parquet \
    --text-column persona_text --persona-role user --min-count 5 \
    --out results/persona_assumption_words.parquet
```

`features` supports `stance`, `sentiment`, `emotion`, and `liwc`. Sentiment and
emotion retain every classifier-label probability. They are not collapsed into
an endorsement axis. LIWC is run directly on the selected text with `pyliwc`;
its output fields are retained with a `liwc_` prefix. `pyliwc` invokes rather
than bundles LIWC, so an installed and activated LIWC application is still
required. Use `--liwc-cli` to select its executable and `--liwc-dictionary` for
a built-in dictionary name or a custom `.dicx` file.

Stored `persona_text` values are chat transcripts. `--persona-role user` is the
default and keeps the person's self-description while excluding the assistant
turns that were generated during persona construction. `assistant` and `all`
are available when those are the intended corpora.

`words` uses Monroe et al.'s Fightin' Words log-odds with an informative prior.
For each assumption label, it contrasts the distinct text units where that
assumption was extracted against matched units where it was not. The command
infers persona-level keys for persona text and response-level keys when prompt
identifiers are available; repeat `--key` to make the unit explicit. The output
is a tidy `(label, word, z, n_target, n_reference)` table. This directly
supports asking which persona words are associated with extracted assumptions,
but it remains an association rather than a causal claim.

The provided long-response file is an extreme-groups sample: its 99 people are
the 50 least and 49 most sycophantic people selected from the binary run, with
no middle group. Descriptive persona or response comparisons on that file must
therefore be reported as comparisons within a deliberately bimodal sample, not
as population rates.

### Reading the join

`join` attaches a per-cell binary score to a parsed assumptions table and ranks
the assumption labels by it. Two levels are available because an assumptions
run and the forced-choice collection may not cover the same grid:

- `cell` -- on (persona_type, persona_id, prompt_id): the sycophancy of the very
  cell whose assumptions these are.
- `persona` -- on (persona_type, persona_id), against that person's mean over
  every dilemma. Coarser, and the level the reference implementation correlates
  at.

`auto` (the default) takes `cell` when it matches anything and falls back.
The match rate of the level taken *and* of the one not taken is printed either
way, because a join that matched 4% of rows looks exactly like one that matched
all of them once it is a mean.

Which framing's assumptions to rank is a real choice, so `join` takes
`--framing`. It defaults to `flipped_story`: that is the framing whose answer
the sycophancy score is about, the probe states its assumptions before that
answer, and it gives exactly one response per design cell. `original_post` asks
the complementary question -- what the model assumed about the person it
correctly absolved. `both` doubles the rows without doubling the information,
because a cell's two responses share one score, so the reported p-values become
optimistic; the command says so when you ask for it.

**The dilemma is the confound, and it is larger than the effect.** Sycophancy
varies far more between dilemmas than between people: in this repo's current
`Gemma3-12B` assumptions run the two dilemmas score 5% and 100%, and the second
has no within-dilemma variance at all. Ranking labels by their raw mean
therefore ranks them by which dilemma they were stated on.
`sycophancy_by_assumption` instead takes the contrast *inside* each dilemma --
`mean(score | label) - mean(score | not label)`, pooled across dilemmas with the
label's count as the weight -- and reports it as `within_delta` beside the raw
mean. `n_strata` counts the dilemmas that contributed, and `n_informative` the
rows of those that varied at all. A warning fires below 20 dilemmas.

**A ranking without a standard error is a ranking of noise.** Several hundred
free-text labels sorted on a difference will always have a striking top and
bottom, so the table also carries `within_se`, `z`, `p_value`, and a
Benjamini-Hochberg `q_value` across the labels reported, and the command prints
how many survive `q < 0.05`. On the current two-dilemma run: **none of them.**
The p-values assume responses are independent given the dilemma; they are not
clustered on the person, which is second-order here only because a label is
usually specific to one dilemma anyway.

**Ranking raw labels is usually the wrong grain.** `summarize` groups the
assumption text by exact normalized string, and this model does not reuse
strings: the current run has **616 distinct labels over 703 responses**, and
only 63 of them are stated five times or more. A per-label sycophancy table is
therefore mostly a table of near-singletons. `--field` exists for that: run
`syco topics` first and rank its topics instead, which for this run collapses
the 616 labels into 50 groups (14.5% left as outliers).

```bash
python -m syco topics results/Gemma3-12B/openended3_assumptions.parquet --seed 0
python -m syco sycophancy join \
    results/Gemma3-12B/openended3_topic_assignments.parquet \
    --binary results/sycophancy/gemma-3-12b-it_binary_sycophancy.parquet \
    --field topic --min-count 10
```

BERTopic's outliers arrive as topic `-1` and are ranked alongside the real
topics; `-1` is "no coherent topic", not a finding, so read past it.

So: **widen the assumptions run before reading the join as a result.**

Finally, an association here is not an effect, and there are two reasons rather
than one. The assumption and the sycophancy are two readings of the same
conditioned model, so a label that co-occurs with sycophancy has not been shown
to cause it -- the intervention the paper uses for that is steering on the
assumption, which this repo does not do. And the two readings do not even come
from the same generation: the assumptions run serves Gemma 3 12B as a 4-bit
GGUF at temperature 0.7 under the paper's third-party probe, while the answer
tables in `files/` were collected in bf16 with greedy decoding under a
first-person prompt. Cells are matched on design coordinates, not on a shared
forward pass, and serving differences are inside the join.

## The three-question analysis

`syco topics` and `syco sycophancy` each answer one question and print it.
`syco analyze` answers the three the study is actually about, over one shared
topic space, and writes the answers as files rather than printing them:

```bash
python -m syco analyze \
    --model Llama-3.1-8B --model Qwen3.6-35B-A3B --model Gemma3-12B \
    --out results/analysis
```

With no `--model` it takes every `results/*/<probe>_assumptions.parquet` it
finds, and warns about any whose grid is less complete than the fullest one.
Sections can be run separately with `--sections 1,2`. Expect roughly half an
hour at the default 2,000 permutations; the embeddings and the fitted topic
model are cached under `<out>/shared/cache`, so a re-run with different test
settings skips both.

The output is a directory: `KEY_FINDINGS.md` for what survived, `report.html`
for the same thing as a self-contained page, a `README.md` per section, and one
CSV per table. Every figure is written twice, light and dark, and has the CSV
holding the same numbers beside it.

### Why there are three different nulls

The three questions have three different units of independence, and using one
test for all of them would be wrong twice.

* **Facet and framing are within-subject.** The same person appears under all
  ten disclosed facets and the same dilemma is told from both sides, so the
  null is not "these labels were drawn from the whole sample" but "inside this
  person and this dilemma, the labels could have been attached to any of these
  responses". The test shuffles the label inside each block. This matters in
  both directions: it stops the between-dilemma effect (which is large) from
  being read as a facet effect, and it is *more* sensitive than the pooled
  test, because the person and the dilemma are differenced out rather than
  left in the residual.

* **A demographic attribute belongs to a person.** Twenty-five personas carry
  about twelve thousand responses between them, and an attribute is constant
  across all of one person's rows. A test that counts responses reports a
  five-decimal p-value off an effective n of 25. The test shuffles the
  person-to-attribute map instead. Both p-values are on the table so the size
  of that gap is visible rather than something to take on trust.

* **Sycophancy is dominated by the dilemma.** Some stories are absolved
  whatever the model thinks it is talking to. The "most" and "least"
  sycophantic thirds are therefore cut inside each dilemma, and the contrast is
  pooled across them.

### Two p-values on every per-cell table

A Monte-Carlo p-value cannot fall below `1 / (draws + 1)`. That is harmless for
a dozen omnibus tests and fatal for a family of several thousand cells: at
2,000 draws the smallest attainable p is 5e-4, and Benjamini-Hochberg over
3,500 tests then cannot produce a q-value below 1 no matter what the data says.
Every cell would report "not significant" for a reason that is about the number
of shuffles drawn.

So each per-cell table carries both. `p_monte_carlo` is the share of shuffles
at least as extreme -- assumption-free, and floored. `p_normal` is the z-score
against the same shuffles' own mean and standard deviation -- unfloored, and
justified because these are counts summed over hundreds of exchangeable blocks.
`q_normal` corrects `p_normal`, and `testable` marks the cells where the
approximation is safe: the null has to vary, and the expected count has to
reach 5. Cells that fail that keep their Monte-Carlo p and an explicit empty
q-value rather than one the approximation cannot support.

### Words and bigrams, and what gets dropped from them

Section 0 is the paper's descriptive pass: `syco.topics.ngram_frequencies` over
the assumption explanations, per persona type, per side of the story, and per
crossing of the two. Beside it, `term_contrasts.csv` gives the z-scored
log-odds of what one condition says *more* than another -- the frequency table
alone cannot separate conditions, because its top is what they all share.

Two things are removed by default and both are switchable.

* **Function words, from bigrams as well as unigrams** (`--stopwords all`).
  The paper keeps them in bigrams -- its own table reports "rather than" and
  "may have" -- but over this corpus that yields "user a", "a is", "may be".
  `--stopwords unigrams` restores the paper's setting.
* **The words the probe supplies** (`syco.topics.PROMPT_ECHO`). The instrument
  addresses the person as "User A", so `user` appears in 99% of the assumptions
  and half the top bigrams were `user <something>`. That is the prompt echoing,
  not the model choosing. `--keep-prompt-echo` leaves them in.

With both applied the top bigrams become "setting boundaries", "personal
space", "high standards", "feeling overwhelmed".

### Which text gets clustered

`--field description` (the default, matching `syco topics` and the paper's own
frequency tables) clusters the explanation the model gave for each mental
model. `--field assumption` clusters the short label alone.

The trade is worth knowing. A description restates the dilemma, so some of its
clusters are one story's furniture -- "the wedding", "the barista", "the hotel
guest" -- which cannot move across dilemmas. A label is the model's
characterization of the person -- "Guilt-Ridden Friend", "Autonomy Seeker" --
so its clusters cut across dilemmas, but it is a much thinner string to
cluster. Either way `shared/topics.csv` carries `dilemma_spread` per topic, so
a story-specific cluster is visible as one, and every contrast in every section
holds the dilemma fixed regardless.

### The demographics file is not used

`files/personas_demographics_vulnerability_final.csv` codes each of the 200
personas for age, gender, education, income, religion, political leaning and
four vulnerability indices. **None of it is swept.** A run draws 25 of those
200 people, so every column lands as two or three groups of eight; a
three-level contrast off eight people per level is not a comparison, and a rank
correlation over 25 points is not an estimate. Adding it back would produce
thirteen more dropdowns of nothing.

What the persona side of the study can support is the transcripts themselves,
and there are 250 of them (25 people x 10 persona types).
`persona_words_by_topic.csv` splits them at the terciles of how often the model
gave that transcript a given topic and contrasts the two thirds' wording with
Fightin' Words. The tercile split rather than presence/absence is forced by the
design: each transcript draws about forty responses, so for any common topic
nearly every persona gets it at least once and there is no comparison group
left. `persona_words_by_assumption.csv` runs the repo's own
`marked_words_by_assumption` at the verbatim-label level, where presence and
absence still both occur.

### Reading the sycophancy section

**Every join is within-model.** Each analyzed model is matched to its own
forced-choice collection, looked for under `results/sycophancy/` and `files/`
by name; `--scores ALIAS=PATH` overrides the match and the resolved mapping is
printed. A model with no collection of its own is **skipped** and named in
`sycophancy_coverage.csv`. Scoring it against another model's numbers would
answer a different question -- which (persona, dilemma) cells invite the
behaviour -- while reading as if it were about the responding model.

The primary outcome is the **continuous** log-odds, not the 0/1 indicator. The
indicator is constant inside all but a handful of dilemmas, so a within-dilemma
contrast on it is estimated off those few and reports a null that belongs to
the threshold. `sycophancy_coverage.csv` puts `dilemmas_binary_varies` next to
`dilemmas_logit_varies` so the difference is on the page. The probability
`sycophancy_soft` is not used at all: it saturates within 1e-8 of 0 or 1, so
its within-dilemma variance is floating-point noise and differences of a
millionth come back at p = 1e-9.

## What's deliberately not here

- **The paper's 5-point social-sycophancy judge.** It is at
  `verbalizedassumptions/elephant_scorer_5pointscale.py` and it is an LLM judge
  over three dimensions, each rated on a five-point scale; the parsed
  `response` column (`--keep-response`) is what it would score. It remains
  unintegrated until its provider configuration and evaluation protocol are
  made reproducible. The text-analysis helpers are not a substitute for it.
- **Reasoning/two-step variants.** The reference also contains `twostep`,
  `ten`, and `supporttypestwostep`. They remain unported rather than being
  approximated from the three character-for-character tested instruments.

## Experiment profiles and orchestration

`config/experiments/default.yaml` owns the probe and design. Its `models:
enabled` selector is resolved from `config/models.yaml`; aliases are not copied
into shell. Per-model `resources.estimated_vram_mib` values also live in the
model registry. The Python scheduler uses `sys.executable`, polls child
processes directly, reaps them immediately, reports GPU-memory waits, supports
`--wait-timeout`, and locks a profile against concurrent writers.

Useful operations:

```bash
python -m syco run --all --limit-per-model 500
python -m syco status --profile default
python -m syco merge --allow-partial       # explicit exploratory partial merge
python -m syco pipeline                    # run, strict merge, parse, summarize
```

API models use the same single-model command. The adapters read credentials
from the provider-standard environment variables and never accept keys as CLI
arguments:

```bash
OPENAI_API_KEY=... python -m syco run --model GPT-5.6 --n-personas 1 --n-prompts 1
ANTHROPIC_API_KEY=... python -m syco run --model Claude-Sonnet-5 \
    --n-personas 1 --n-prompts 1
```

GPT-5.6 uses OpenAI's Responses API with `reasoning.effort` and
`max_output_tokens`; unsupported `temperature`/`top_p` fields are omitted.
Claude Sonnet 5 uses explicit disabled thinking with low output effort for the
study's default thinking-off condition and likewise omits unsupported sampling
fields. Provider 4xx validation/authentication errors fail immediately, while
rate limits, timeouts, and server errors remain retryable.
