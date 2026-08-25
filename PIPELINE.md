# Open-ended verbalized assumptions

Runs the open-ended assumptions probe from Chen et al., *Verbalizing LLMs'
assumptions to explain and control sycophancy* (vendored under
[verbalizedassumptions/](verbalizedassumptions/)) over this repo's persona x
dilemma design, through the model layer from
[myazan/LLM-Self-Concept](https://github.com/myazan/LLM-Self-Concept).

## The probe

Before answering, the model states its top-k mental models of the user with
probabilities, then replies under a `RESPONSE:` heading. Assumptions come
**first** — that ordering is the method. An assumption written after the answer
would be a rationalization of a reply already committed to.

The question this design puts to it: the same person, the same dilemma, ten
different facets of them disclosed. What the model says it is talking to, and
how that moves, is the measurement.

## The design

| factor | levels | source |
|---|---|---|
| `persona_type` | 10 facets (hobbies, politics, family, …) + no-persona control | `files/base_data_persona.gz` |
| `persona_id` | 200 synthetic people, each appearing once per facet | same |
| `prompt_type` | `original_post`, `flipped_story` — the same dilemma from either side | `files/base_data_prompt.gz` |
| `prompt_id` | 1000 dilemmas | same |

Fully crossed that is 4M cells, so every run takes a subset — and subsetting is
always **paired**: personas and dilemmas are drawn once, then crossed with all
the levels. Both contrasts stay within-subject that way. Sampling independently
per condition would confound the contrast with whichever personas happened to be
drawn, and nothing downstream recovers from that.

## Two ways to send the conversation

- `--history-mode native` (default) — the persona transcript goes as **real chat
  turns** and the probe rides on the final user message. This is how
  `files/*_long_results.pkl` was collected, so an assumption row and its
  existing `model_answer` share a prompt prefix and compare cell for cell.
- `--history-mode inline` — the paper's own shape: the transcript flattened into
  a `Conversation so far: """..."""` block, model addressed as a third party
  observing "User A". A robustness check on the same cells. Expect it to differ:
  a model reading a transcript is in a different position from one that has been
  in the conversation.

## Run it

```bash
pip install -r requirements.txt
python -m syco.model_registry          # what's configured, and how each is served

# plan a run — no weights, no keys, no cost
python scripts/run_assumptions.py --model Gemma3-12B --plan-only \
    --n-personas 20 --n-prompts 25

# exercise the whole pipeline offline against the mock backend
python scripts/run_assumptions.py --model Gemma3-12B --dry-run \
    --n-personas 2 --n-prompts 2 --out results/smoke.jsonl

# the real thing, on cells the existing answers table already covers
python scripts/run_assumptions.py --model Gemma3-12B \
    --match-existing files/gemma-3-12b-it_long_results.pkl \
    --n-personas 25 --n-prompts 20 \
    --out results/gemma3-12b_openended.jsonl

python scripts/parse_assumptions.py    results/gemma3-12b_openended.jsonl
python scripts/summarize_assumptions.py results/gemma3-12b_openended_assumptions.parquet
```

Runs resume by default — re-run the same command with the same `--out`. Rows
that recorded an error are retried; a truncated final line from a killed run is
ignored. Ctrl-C once finishes the batch in flight and flushes.

`--match-existing` is the flag that matters for comparing against work already
done: it restricts the grid to the `(persona_id, prompt_id)` pairs a prior
answers table covers, so every assumption row lands beside an answer from the
same model on the same cell.

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

`run_assumptions.py` writes JSONL, one row per cell, holding the completion
**verbatim** plus the design columns and full model provenance. Parsing is a
separate step so a parser fix is a re-parse, not a re-run.

`parse_assumptions.py` writes two tables and prints the instrument's health
check. Read that first: `clean` / `repaired` / `salvaged` / `failed` per persona
facet. A facet that parses worse than the others differs in format compliance,
and that difference will masquerade as a finding in everything downstream.

## What's deliberately not here

- **Grouping the assumption labels.** They are free text, currently grouped by
  normalized string, so near-synonyms ("wants validation" / "seeking
  validation") sit in separate rows. Clustering or embedding them is the next
  step and a consequential one — how the labels are grouped decides what the
  frequency tables say.
- **Sycophancy scoring of the replies.** The paper's 5-point social sycophancy
  judge is at `verbalizedassumptions/elephant_scorer_5pointscale.py`; the parsed
  `response` column (`--keep-response`) is what it would score.
- **The other probes.** `get_assumptions.py` also has `4dims`, `supporttypes`
  and two-step variants. `syco/prompts.py` is where they would go — the runner
  takes whatever `ProbeSpec` builds.
