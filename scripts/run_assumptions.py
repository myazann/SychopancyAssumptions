#!/usr/bin/env python3
"""Administer the open-ended verbalized-assumptions probe over the persona x
dilemma design.

The probe is from Chen et al., *Verbalizing LLMs' assumptions to explain and
control sycophancy*: before answering, the model states its top-k mental models
of the user with probabilities, then replies. Asking for the assumptions FIRST
is the method -- it makes visible what the model thinks it is talking to, which
is exactly the thing a persona is supposed to be moving.

Examples
--------
Plan a run without loading a model (no weights, no keys, no cost):

    python -m syco run --model Gemma3-12B --plan-only \
        --n-personas 20 --n-prompts 25

Exercise the whole pipeline offline, including the parser:

    python -m syco run --model Gemma3-12B --dry-run \
        --n-personas 2 --n-prompts 2 --out results/smoke.jsonl

Collect assumptions for cells the existing answers table already covers, so
every assumption row sits beside an answer from the same model:

    python -m syco run --model Gemma3-12B \
        --match-existing files/gemma-3-12b-it_long_results.pkl \
        --n-personas 25 --n-prompts 20 \
        --out results/gemma3-12b_openended.jsonl

The paper's own framing, as a robustness check on the same cells:

    python -m syco run --model Gemma3-12B --history-mode inline \
        --out results/gemma3-12b_openended_inline.jsonl

Runs resume by default: re-running the same command with the same --out picks
up where it stopped, and rows that recorded an error are retried.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from syco import grid
from syco import prompts as probe_prompts
from syco.data import load_answers, load_personas, load_prompts
from syco.manifest import build_manifest, ensure_manifest, manifest_path
from syco.model_registry import ANTHROPIC_BACKEND, load_registry
from syco.models import Conversation, build_adapter
from syco.paths import RESULTS_DIR
from syco.store import AssumptionRecord, append, completed_keys

# Records buffered before an fsync'd append -- bounds worst-case loss on a hard
# kill without fsyncing every single cell.
FLUSH_EVERY = 25
# ...except at the very start, where the first few cells are flushed immediately.
# The first thing anyone does with a new run is check that the rows look right,
# and waiting 25 cells for that on a slow local model is the difference between
# catching a bad prompt in a minute and catching it in an hour.
FLUSH_FIRST = 3


class GracefulKiller:
    """First Ctrl-C finishes the batch in flight and flushes; second one exits."""

    def __init__(self):
        self.stop = False
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self._handle)

    def _handle(self, signum, frame):
        if self.stop:
            print("\nsecond interrupt -- exiting now", flush=True)
            raise KeyboardInterrupt
        self.stop = True
        print("\ninterrupt received -- finishing the batch in flight, then "
              "flushing. Ctrl-C again to drop it.", flush=True)


def fmt_duration(seconds) -> str:
    if seconds is None:
        return "?"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


def build_conversation(cell, spec_probe, system: str) -> Conversation:
    messages = probe_prompts.build(spec_probe, cell.persona.messages, cell.prompt.text)
    return Conversation(messages=tuple(messages), system=system)


def make_record(
    cell,
    spec,
    spec_probe,
    plan,
    key,
    raw: str,
    conv,
    run_id: str,
    error: str = "",
    backend_override: str | None = None,
) -> AssumptionRecord:
    provenance = spec.provenance()
    thinking_type = (plan.kwargs.get("thinking") or {}).get("type")
    if spec.backend == ANTHROPIC_BACKEND and thinking_type in {"adaptive", "enabled"}:
        provenance["temperature"] = None
    if backend_override == "mock":
        provenance.update(temperature=None, top_p=None)
    # These values are constant for a per-model run and already live in the
    # adjacent manifest. Repeating them in every row wastes space and obscures
    # the actual design and observation columns.
    for column in (
        "model_id", "model_family", "model_generation", "model_release_date",
        "backend", "quantized_file",
    ):
        provenance.pop(column, None)
    return AssumptionRecord(
        cell_key=key,
        persona_type=cell.persona.persona_type,
        persona_id=cell.persona.persona_id,
        prompt_type=cell.prompt.prompt_type,
        prompt_id=cell.prompt.prompt_id,
        rep=cell.rep,
        probe=spec_probe.label(),
        history_mode=spec_probe.history_mode,
        n_assumptions_asked=spec_probe.n_models,
        persona_turns=cell.persona.n_turns,
        persona_recovered=cell.persona.recovered,
        thinking_applied=plan.applied,
        thinking_standardized=plan.standardized,
        raw=raw,
        prompt_digest=conv.digest(),
        error=error,
        timestamp=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        run_id=run_id,
        **provenance,
    )


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def nonnegative_float(value: str) -> float:
    number = float(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return number


def probability(value: str) -> float:
    number = float(value)
    if not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return number


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--model", required=True,
                   help="alias from config/models.yaml (see `python -m syco.model_registry`)")
    p.add_argument("--out", default=None,
                   help="results JSONL (default: results/<alias>_<probe>.jsonl)")

    g = p.add_argument_group("instrument")
    g.add_argument("--probe", choices=("openended", "plain"), default="openended",
                   help="openended = assumptions then reply; plain = reply only "
                        "(the no-probe control, matching the existing answers table)")
    g.add_argument("--history-mode", choices=probe_prompts.HISTORY_MODES,
                   default=probe_prompts.NATIVE,
                   help="native: persona as real chat turns (default, matches how "
                        "the answers table was collected). inline: the paper's "
                        "transcript-in-one-message framing")
    g.add_argument("--n-models", type=positive_int,
                   default=probe_prompts.DEFAULT_N_MODELS,
                   help="how many mental models to ask for (paper: 3)")
    g.add_argument("--system", default="",
                   help="system prompt, applied to every cell (default: none)")

    g = p.add_argument_group("design")
    g.add_argument("--persona-types", nargs="*", default=None,
                   help="facets to include (default: all ten)")
    g.add_argument("--prompt-types", nargs="*", default=None,
                   choices=("original_post", "flipped_story"),
                   help="framings to include (default: original_post flipped_story)")
    g.add_argument("--n-personas", type=positive_int, default=None,
                   help="people to sample; the SAME people appear in every facet")
    g.add_argument("--n-prompts", type=positive_int, default=None,
                   help="dilemmas to sample; both framings of each are kept")
    g.add_argument("--n-reps", type=positive_int, default=1,
                   help="draws per cell (>1 measures within-cell variance)")
    g.add_argument("--no-control", action="store_true",
                   help="drop the persona-free control cells")
    g.add_argument("--match-existing", default=None,
                   help="restrict to exact persona/facet/framing/dilemma cells "
                        "in a prior answers table, e.g. "
                        "files/gemma-3-12b-it_long_results.pkl")
    g.add_argument("--seed", type=int, default=1000,
                   help="sampling seed; the same seed gives the same subset")

    g = p.add_argument_group("execution")
    g.add_argument("--limit", type=positive_int, default=None, help="stop after N cells")
    g.add_argument("--batch-size", type=positive_int, default=None,
                   help="local backends: prompts per generate() call")
    g.add_argument("--max-workers", type=positive_int, default=None,
                   help="API backends: concurrent requests")
    g.add_argument("--device", default=None,
                   help="hf backend: override device_map (cuda | cpu | mps | auto). "
                        "mps is never chosen automatically -- Apple's backend "
                        "returns NaN logits for left-padded batches, so pair it "
                        "with --batch-size 1")
    g.add_argument("--temperature", type=nonnegative_float, default=None)
    g.add_argument("--top-p", type=probability, default=None)
    g.add_argument("--max-tokens", type=positive_int, default=None,
                   help="output budget per cell (must fit the JSON block AND the reply)")
    g.add_argument("--thinking", action="store_true",
                   help="let the model reason before answering (default: asserted "
                        "off, so the verbalized block is the only reasoning shown)")
    g.add_argument("--no-resume", action="store_true",
                   help="require an empty/new --out instead of resuming")
    g.add_argument("--overwrite", action="store_true",
                   help="replace --out and its manifest before running")
    g.add_argument("--plan-only", action="store_true",
                   help="print the grid and exit, loading nothing")
    g.add_argument("--dry-run", action="store_true",
                   help="run against the mock backend: no weights, no keys, no cost")
    return p.parse_args(argv)


def configured_spec(args, registry, *, resolve_quant: bool = False):
    """Apply CLI generation/runtime overrides exactly once.

    Orchestration uses the same function to compute the manifest expected for
    the current profile, preventing validation logic from drifting away from
    the actual runner.
    """
    from dataclasses import replace

    spec = registry.get(args.model)
    overrides = {
        key: value
        for key, value in (
            ("temperature", args.temperature),
            ("top_p", args.top_p),
            ("max_output_tokens", args.max_tokens),
            ("batch_size", args.batch_size),
            ("max_workers", args.max_workers),
        )
        if value is not None
    }
    if overrides:
        spec = replace(spec, **overrides)
    if args.device:
        spec = replace(spec, runtime={**spec.runtime, "device_map": args.device})
    if (
        resolve_quant
        and spec.quantization.format == "gguf"
        and not args.dry_run
        and not args.plan_only
    ):
        spec = registry.with_resolved_quant(spec)
    return spec


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.overwrite and args.no_resume:
        raise SystemExit("--overwrite and --no-resume are mutually exclusive")
    if args.overwrite and args.plan_only:
        raise SystemExit("--overwrite cannot be used with --plan-only")

    # Line-buffer stdout: a run of this length is normally redirected to a log,
    # and Python block-buffers a redirected stream, so progress would sit
    # invisible in a 4KB buffer for however long the first cells take.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass

    registry = load_registry()
    spec = configured_spec(args, registry, resolve_quant=True)

    spec_probe = probe_prompts.ProbeSpec(
        kind=args.probe, history_mode=args.history_mode, n_models=args.n_models)

    out = args.out or str(RESULTS_DIR / f"{spec.safe_dir_name()}_"
                          f"{spec_probe.label().replace('/', '-')}.jsonl")

    run_manifest = build_manifest(
        args=args,
        spec=spec,
        probe=spec_probe,
        resolved_file=spec.quantization.resolved_file,
    )
    run_id = run_manifest["run_id"]

    # -- the grid ----------------------------------------------------------
    personas, diag = load_personas()
    dilemmas = load_prompts()
    restrict = None
    if args.match_existing:
        answers = load_answers(args.match_existing)
        restrict = grid.cells_from_answers(answers, args.persona_types)
        print(f"matching {len(restrict)} full design cell(s) from "
              f"{args.match_existing}")

    cells = grid.build_cells(
        personas, dilemmas,
        persona_types=args.persona_types,
        prompt_types=args.prompt_types,
        n_persona_ids=args.n_personas,
        n_prompt_ids=args.n_prompts,
        include_no_persona=not args.no_control,
        n_reps=args.n_reps,
        seed=args.seed,
        restrict_cells=restrict,
    )

    print(f"model:   {spec.alias}  ({spec.backend}, {spec.quantization.label}, "
          f"T={spec.temperature}, top_p={spec.top_p}, max_tokens={spec.max_output_tokens})")
    print(f"probe:   {spec_probe.label()}")
    print(f"run:     {run_id}")
    print(grid.summarize(cells))
    unusable = int((~diag.usable).sum()) if len(diag) else 0
    recovered = int(diag.recovered.sum()) if len(diag) else 0
    if unusable:
        print(f"note:    {unusable} persona transcript(s) unusable and excluded")
    if recovered:
        print(f"note:    {recovered} transcript(s) needed salvage parsing -- "
              "flagged per row as persona_recovered")

    output_path = Path(out)
    if not args.plan_only:
        has_output = output_path.is_file() and output_path.stat().st_size > 0
        if args.no_resume and has_output:
            print(f"error: --no-resume requires a new or empty --out: {out}",
                  file=sys.stderr)
            return 2
        if args.overwrite:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("", encoding="utf-8")
            manifest_path(out).unlink(missing_ok=True)
            has_output = False
        try:
            ensure_manifest(out, run_manifest, has_output=has_output)
        except RuntimeError as err:
            print(f"error: {err}", file=sys.stderr)
            return 2

    done = set() if args.no_resume else completed_keys(out)
    todo = [
        c for c in cells
        if grid.cell_key(spec.alias, spec_probe.label(), c, run_id) not in done
    ]
    if done:
        print(f"resume:  {len(cells) - len(todo)} of {len(cells)} cells already in {out}")
    if args.limit:
        todo = todo[:args.limit]
    print(f"to run:  {len(todo)} cell(s) -> {out}")

    if args.plan_only:
        return 0
    if not todo:
        print("nothing to do.")
        return 0

    # -- administer --------------------------------------------------------
    killer = GracefulKiller()
    started = time.monotonic()
    buffer, written, errors = [], 0, 0

    def flush():
        nonlocal buffer
        if buffer:
            append(buffer, out)
            buffer = []

    def progress(i):
        elapsed = time.monotonic() - started
        rate = written / elapsed if elapsed else 0.0
        eta = (len(todo) - i) / rate if rate > 0 else None
        print(f"  {i}/{len(todo)} | {rate:.2f} cells/s | ETA {fmt_duration(eta)} "
              f"| errors={errors}", flush=True)

    if spec.quantization.format == "gguf" and not args.dry_run:
        print(f"quant file: {spec.quantization.resolved_file}")

    with build_adapter(spec, dry_run=args.dry_run) as adapter:
        plan = adapter.thinking_plan(want_thinking=args.thinking)
        print(f"thinking: {plan.applied}"
              f"{'' if plan.standardized else '  (NOT standardized -- recorded per row)'}")

        def run_one(cell):
            """-> (record). Never raises: a failed cell is a row with an error,
            so the run continues and that cell is retried next time."""
            key = grid.cell_key(spec.alias, spec_probe.label(), cell, run_id)
            conv = build_conversation(cell, spec_probe, args.system)
            try:
                raw = adapter.chat(conv, n=1, plan=plan)[0]
                return make_record(
                    cell,
                    spec,
                    spec_probe,
                    plan,
                    key,
                    raw,
                    conv,
                    run_id,
                    backend_override="mock" if args.dry_run else None,
                )
            except Exception as err:
                return make_record(
                    cell,
                    spec,
                    spec_probe,
                    plan,
                    key,
                    "",
                    conv,
                    run_id,
                    error=f"{type(err).__name__}: {err}",
                    backend_override="mock" if args.dry_run else None,
                )

        if spec.is_api:
            # API: concurrency is the throughput lever, and the provider is the
            # one batching under the hood.
            with ThreadPoolExecutor(max_workers=spec.max_workers) as pool:
                for i, record in enumerate(pool.map(run_one, todo), start=1):
                    buffer.append(record)
                    written += 1
                    errors += bool(record.error)
                    if len(buffer) >= FLUSH_EVERY or written <= FLUSH_FIRST:
                        flush()
                        progress(i)
                    if killer.stop:
                        break
        elif adapter.batches:
            # Local + batching: one generate() per batch. Padding cost is lowest
            # when prompts in a batch are similar lengths, so sort by transcript
            # length -- adjacent cells then pad against each other.
            ordered = sorted(todo, key=lambda c: (len(c.prompt.text), c.persona.n_turns))
            for start in range(0, len(ordered), spec.batch_size):
                chunk = ordered[start:start + spec.batch_size]
                keys = [
                    grid.cell_key(spec.alias, spec_probe.label(), c, run_id)
                    for c in chunk
                ]
                convs = [build_conversation(c, spec_probe, args.system) for c in chunk]
                try:
                    raws = adapter.chat_batch(convs, plan=plan)
                    err = ""
                except Exception as e:
                    raws = [""] * len(chunk)
                    err = f"{type(e).__name__}: {e}"
                    print(f"  [batch error] {err}", flush=True)
                for cell, key, conv, raw in zip(chunk, keys, convs, raws):
                    buffer.append(make_record(
                        cell,
                        spec,
                        spec_probe,
                        plan,
                        key,
                        raw,
                        conv,
                        run_id,
                        error=err,
                        backend_override="mock" if args.dry_run else None,
                    ))
                written += len(chunk)
                errors += len(chunk) if err else 0
                flush()
                progress(min(start + len(chunk), len(ordered)))
                if killer.stop:
                    break
        else:
            for i, cell in enumerate(todo, start=1):
                record = run_one(cell)
                buffer.append(record)
                written += 1
                errors += bool(record.error)
                if len(buffer) >= FLUSH_EVERY or written <= FLUSH_FIRST:
                    flush()
                    progress(i)
                if killer.stop:
                    break
        flush()

    elapsed = time.monotonic() - started
    print(f"\nwrote {written} row(s) in {fmt_duration(elapsed)} "
          f"({errors} error(s)) -> {out}")
    if killer.stop:
        print("stopped early -- re-run the same command to continue.")
    print(f"next: python -m syco parse {out}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
