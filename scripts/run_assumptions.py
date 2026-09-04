#!/usr/bin/env python3
"""Administer a Verbalized Assumptions probe over the persona x dilemma design.

The probes are from Cheng et al., *Verbalizing LLMs' assumptions to explain and
control sycophancy*. Select `openended`, `4dims`, or `supporttypes`; all use the
same paired persona x dilemma grid. Before answering, the model states either
its top-k mental models or its scores on fixed belief dimensions, then replies.
Asking for the assumptions FIRST is the method -- it makes visible what the
model thinks it is talking to, which is exactly the thing a persona is supposed
to be moving.

Examples
--------
Plan a run without loading a model (no weights, no keys, no cost):

    python -m syco run --model Gemma3-12B --plan-only \
        --n-personas 20 --n-prompts 25

Exercise the whole pipeline offline, including the parser:

    python -m syco run --model Gemma3-12B --dry-run \
        --n-personas 2 --n-prompts 2 --out results/smoke.jsonl

Run the four structured sycophancy-related dimensions:

    python -m syco run --model Gemma3-12B --probe 4dims --dry-run \
        --n-personas 2 --n-prompts 2

Collect assumptions for cells the existing answers table already covers, so
every assumption row sits beside an answer from the same model:

    python -m syco run --model Gemma3-12B \
        --n-personas 25 --n-prompts 20 \
        --out results/gemma3-12b_openended.jsonl

Runs resume by default: re-running the same command with the same --out picks
up where it stopped, and rows that recorded an error are retried.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import os
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from syco import grid
from syco import prompts as probe_prompts
from syco.data import CONTROL_PERSONA, load_personas, load_prompts
from syco.extensions import cell_coordinate, plan_extension, validate_compatible_run
from syco.manifest import build_manifest, manifest_path, reconcile_manifest
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
        print(
            "\ninterrupt received -- finishing the batch in flight, then "
            "flushing. Ctrl-C again to drop it.",
            flush=True,
        )


def fmt_duration(seconds) -> str:
    if seconds is None:
        return "?"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


def build_conversation(
    cell,
    spec_probe,
    system: str,
    *,
    four_dims_prompt_version: str = probe_prompts.FOUR_DIMS_PROMPT_EXPLICIT_V2,
) -> Conversation:
    messages = probe_prompts.build(
        spec_probe,
        cell.persona.messages,
        cell.prompt.text,
        four_dims_prompt_version=four_dims_prompt_version,
    )
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
        "model_id",
        "model_family",
        "model_generation",
        "model_release_date",
        "backend",
        "quantized_file",
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
        n_assumptions_asked=(
            spec_probe.n_models if spec_probe.family == "open-ended" else 0
        ),
        n_dimensions_asked=len(spec_probe.dimensions),
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
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    p.add_argument(
        "--model",
        required=True,
        help="alias from config/models.yaml (see `python -m syco.model_registry`)",
    )
    p.add_argument(
        "--out",
        default=None,
        help="results JSONL (default: results/<alias>/<probe>.jsonl)",
    )

    g = p.add_argument_group("instrument")
    g.add_argument(
        "--probe",
        choices=probe_prompts.PROBE_KINDS,
        default="openended",
        help="paper probe to administer: openended, 4dims, or "
        "supporttypes (default: openended)",
    )
    g.add_argument(
        "--n-models",
        type=positive_int,
        default=probe_prompts.DEFAULT_N_MODELS,
        help="openended only: how many mental models to ask for "
        "(paper: 3; ignored by structured probes)",
    )
    g.add_argument(
        "--system",
        default="",
        help="system prompt, applied to every cell (default: none)",
    )
    g.add_argument(
        "--four-dims-prompt-version",
        choices=probe_prompts.FOUR_DIMS_PROMPT_VERSIONS,
        default=probe_prompts.FOUR_DIMS_PROMPT_EXPLICIT_V2,
        help="4dims prompt wording; paper-v1 exists only to resume legacy runs",
    )

    g = p.add_argument_group("design")
    g.add_argument(
        "--persona-types",
        nargs="*",
        default=None,
        help="facets to include (default: all ten)",
    )
    g.add_argument(
        "--prompt-types",
        nargs="*",
        default=None,
        choices=("original_post", "flipped_story"),
        help="framings to include (default: original_post flipped_story)",
    )
    g.add_argument(
        "--n-personas",
        type=positive_int,
        default=None,
        help="people to sample; the SAME people appear in every facet",
    )
    g.add_argument(
        "--n-prompts",
        type=positive_int,
        default=None,
        help="dilemmas to sample; both framings of each are kept",
    )
    g.add_argument(
        "--n-reps",
        type=positive_int,
        default=1,
        help="draws per cell (>1 measures within-cell variance)",
    )
    g.add_argument(
        "--no-control", action="store_true", help="drop the persona-free control cells"
    )
    g.add_argument(
        "--seed",
        type=int,
        default=1000,
        help="sampling seed; the same seed gives the same subset",
    )
    g.add_argument(
        "--design",
        type=Path,
        default=None,
        help="frozen study design; uses its exact IDs and verifies source data",
    )
    g.add_argument(
        "--extend-from",
        action="append",
        default=None,
        metavar="JSONL",
        help="an acquisition shard already collected; repeat for each earlier "
        "wave. This run emits only the cells the target design still lacks. "
        "With --design the target is that design; without it, "
        "--n-personas/--n-prompts mean additional disjoint IDs drawn off one "
        "complete base",
    )

    g = p.add_argument_group("execution")
    g.add_argument(
        "--limit", type=positive_int, default=None, help="stop after N cells"
    )
    g.add_argument(
        "--batch-size",
        type=positive_int,
        default=None,
        help="local backends: prompts per generate() call",
    )
    g.add_argument(
        "--max-workers",
        type=positive_int,
        default=None,
        help="API backends: concurrent requests",
    )
    g.add_argument(
        "--device",
        default=None,
        help="hf backend: override device_map (cuda | cpu | mps | auto). "
        "mps is never chosen automatically -- Apple's backend "
        "returns NaN logits for left-padded batches, so pair it "
        "with --batch-size 1",
    )
    g.add_argument("--temperature", type=nonnegative_float, default=None)
    g.add_argument("--top-p", type=probability, default=None)
    g.add_argument(
        "--max-tokens",
        type=positive_int,
        default=None,
        help="output budget per cell (must fit the JSON block AND the reply)",
    )
    g.add_argument(
        "--thinking",
        action="store_true",
        help="let the model reason before answering (default: asserted "
        "off, so the verbalized block is the only reasoning shown)",
    )
    g.add_argument(
        "--no-resume",
        action="store_true",
        help="require an empty/new --out instead of resuming",
    )
    g.add_argument(
        "--overwrite",
        action="store_true",
        help="replace --out and its manifest before running",
    )
    g.add_argument(
        "--plan-only",
        action="store_true",
        help="print the grid and exit, loading nothing",
    )
    g.add_argument(
        "--dry-run",
        action="store_true",
        help="run against the mock backend: no weights, no keys, no cost",
    )
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


def verify_prompt_digests(
    rows,
    personas,
    prompts,
    spec_probe,
    system,
    *,
    four_dims_prompt_version=probe_prompts.FOUR_DIMS_PROMPT_EXPLICIT_V2,
    limit=None,
):
    """Prove that current prompt construction still reproduces stored prompts.

    This is the check that makes a long, interrupted study safe to continue.
    Rather than requiring the source tree to be byte-identical to the one that
    collected earlier rows -- which no study running for weeks can promise --
    it rebuilds the prompt for rows already on disk and compares the digest the
    run recorded. If the prompts reproduce, the old and new rows are the same
    observation regardless of what else changed in the repository.

    `limit` samples evenly across `rows` instead of checking all of them, for
    the resume path where the file may hold tens of thousands of rows.
    """
    persona_by_key = {
        (persona.persona_type, persona.persona_id): persona for persona in personas
    }
    persona_by_key[(CONTROL_PERSONA.persona_type, CONTROL_PERSONA.persona_id)] = (
        CONTROL_PERSONA
    )
    prompt_by_key = {
        (prompt.prompt_type, prompt.prompt_id): prompt for prompt in prompts
    }
    rows = [row for row in rows if not row.get("error")]
    if limit is not None and len(rows) > limit:
        step = len(rows) / limit
        rows = [rows[int(index * step)] for index in range(limit)]
    mismatches = []
    for row in rows:
        stored = row.get("prompt_digest")
        if not stored:
            raise RuntimeError(
                "a prior row carries no prompt_digest; prompt compatibility "
                "cannot be verified"
            )
        persona_key = (str(row.get("persona_type")), str(row.get("persona_id")))
        prompt_key = (str(row.get("prompt_type")), str(row.get("prompt_id")))
        try:
            cell = grid.Cell(
                persona=persona_by_key[persona_key],
                prompt=prompt_by_key[prompt_key],
                rep=int(row.get("rep", 0)),
            )
        except KeyError as err:
            raise RuntimeError(
                f"a prior coordinate is absent from current source data: {err}"
            ) from err
        current = build_conversation(
            cell,
            spec_probe,
            system,
            four_dims_prompt_version=four_dims_prompt_version,
        ).digest()
        if current != stored:
            mismatches.append(row.get("cell_key") or str(prompt_key))
            if len(mismatches) >= 5:
                break
    if mismatches:
        raise RuntimeError(
            "current prompt construction differs from rows already collected; "
            f"first mismatched cells: {', '.join(mismatches)}"
        )


RESUME_DIGEST_SAMPLE = 40


@contextlib.contextmanager
def single_writer(out):
    """Hold an advisory lock on an output for the life of the run.

    Two writers appending to one JSONL is the one failure this pipeline cannot
    repair after the fact: the rows interleave, and the duplicate attempts look
    like ordinary retries. `syco run` is invoked directly by every Slurm array
    task, so the profile-level lock in `orchestrate` never sees them.

    Advisory locking is not available on every shared filesystem. Where it is
    missing the run proceeds with a warning rather than refusing to start --
    losing the lock is bad, losing a queued 120-hour job is worse.
    """
    import fcntl

    path = Path(f"{out}.lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = path.open("a+")
    except OSError as err:
        print(f"note:    cannot open {path} ({err}); running without a writer lock")
        yield
        return
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(
                f"another run is already writing {out}. Wait for it to finish, or "
                "choose a different --out; two writers on one JSONL cannot be "
                "untangled afterwards."
            ) from None
        except OSError as err:
            print(
                f"note:    {path} does not support locking ({err}); "
                "running without a writer lock"
            )
        yield
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def check_resumable(out, cells, run_id, personas, prompts, spec_probe, args) -> None:
    """Refuse to continue an output whose rows this plan would not have produced.

    Two things have to hold before appending to an existing shard: every row
    already there must be a cell this configuration still administers, and the
    prompts must still build the same way. Together those are what `run_id`
    equality used to stand in for -- but they are checked against the data
    instead of against a hash of the source tree, so a study can be continued
    after the analysis code around it has moved on.
    """
    from syco.store import canonical_rows, read_rows

    rows, _ = canonical_rows(read_rows(out))
    if not rows:
        return
    planned = {cell_coordinate(cell) for cell in cells}
    from syco.extensions import row_coordinate

    stray = {row_coordinate(row) for row in rows} - planned
    if stray:
        example = ", ".join("/".join(str(p) for p in c) for c in sorted(stray)[:3])
        raise RuntimeError(
            f"{out} holds {len(stray)} cell(s) this configuration no longer "
            f"administers (e.g. {example}). The design shrank or changed; choose "
            "a new --out rather than mixing designs in one file."
        )
    foreign = [row for row in rows if row.get("run_id") not in (run_id, None, "")]
    if foreign:
        raise RuntimeError(
            f"{out} holds {len(foreign)} row(s) from another run; refusing to append"
        )
    verify_prompt_digests(
        rows,
        personas,
        prompts,
        spec_probe,
        args.system,
        four_dims_prompt_version=args.four_dims_prompt_version,
        limit=RESUME_DIGEST_SAMPLE,
    )


@dataclass(frozen=True)
class RunPlan:
    """What a configuration administers, resolved once and reused everywhere.

    The runner, `syco status`, and `syco merge` all need this answer, and when
    they each derived it themselves they drifted apart: orchestration kept
    passing raw `--persona-types` while the runner had already rewritten them
    from the frozen design, so status reported every extension as unplannable.
    One function, three callers.
    """

    cells: tuple
    diagnostics: object
    frozen_design: dict | None
    extension: object | None
    personas: tuple
    prompts: tuple

    @property
    def coordinates(self) -> list:
        return [cell_coordinate(cell) for cell in self.cells]


def apply_frozen_design(args, spec_probe):
    """Resolve `--design` into `args`, so every caller plans the same grid."""
    if not args.design:
        return None
    from syco.design import selection_for
    from syco.paths import PERSONA_PATH, PROMPT_PATH

    frozen = selection_for(
        args.design,
        probe=spec_probe.kind,
        persona_path=PERSONA_PATH.resolve(strict=True),
        prompt_path=PROMPT_PATH.resolve(strict=True),
    )
    expected_instrument = {
        "probe": spec_probe.kind,
        "family": spec_probe.family,
        "n_models": spec_probe.n_models if spec_probe.family == "open-ended" else None,
        "dimensions": list(spec_probe.dimensions),
        "system": args.system,
        "thinking": bool(args.thinking),
    }
    for key, expected in expected_instrument.items():
        recorded = frozen["instrument"].get(key)
        if recorded != expected:
            raise RuntimeError(
                f"--design fixes instrument {key} as {recorded!r}; got {expected!r}"
            )
    factors = frozen["factors"]
    for attribute in ("persona_types", "prompt_types"):
        requested = getattr(args, attribute)
        expected = list(factors[attribute])
        if requested is not None and list(requested) != expected:
            raise RuntimeError(
                f"--design fixes {attribute.replace('_', '-')} as {expected}; "
                f"got {requested}"
            )
        setattr(args, attribute, expected)
    if args.n_reps != int(factors["n_reps"]):
        raise RuntimeError(
            f"--design fixes n-reps as {factors['n_reps']}; got {args.n_reps}"
        )
    expected_control = bool(factors["include_control"])
    if args.no_control and expected_control:
        raise RuntimeError("--design includes control cells; remove --no-control")
    args.no_control = not expected_control
    return frozen


def build_plan(args, spec_probe, *, personas=None, prompts=None) -> RunPlan:
    """Resolve a configuration into the exact cells it administers.

    Mutates `args` where a frozen design fixes a value, so the manifest records
    what was administered rather than what was typed.
    """
    diagnostics = None
    if personas is None:
        personas, diagnostics = load_personas()
    if prompts is None:
        prompts = load_prompts()
    frozen = apply_frozen_design(args, spec_probe)

    extension_plan = None
    if args.extend_from:
        if frozen is None and (args.n_personas is None or args.n_prompts is None):
            raise RuntimeError(
                "--extend-from requires --design or both --n-personas and --n-prompts"
            )
        extension_plan = plan_extension(
            args.extend_from,
            personas,
            prompts,
            persona_types=args.persona_types,
            prompt_types=args.prompt_types,
            additional_personas=args.n_personas,
            additional_prompts=args.n_prompts,
            include_no_persona=not args.no_control,
            n_reps=args.n_reps,
            seed=args.seed,
            target_persona_ids=frozen["selection"]["persona_ids"] if frozen else None,
            target_prompt_ids=frozen["selection"]["prompt_ids"] if frozen else None,
        )
        cells = list(extension_plan.cells)
    else:
        if frozen and (args.n_personas is not None or args.n_prompts is not None):
            raise RuntimeError(
                "--design supplies exact IDs; do not also pass sample counts"
            )
        cells = grid.build_cells(
            personas,
            prompts,
            persona_types=args.persona_types,
            prompt_types=args.prompt_types,
            n_persona_ids=args.n_personas,
            n_prompt_ids=args.n_prompts,
            include_no_persona=not args.no_control,
            n_reps=args.n_reps,
            seed=args.seed,
            persona_ids=frozen["selection"]["persona_ids"] if frozen else None,
            prompt_ids=frozen["selection"]["prompt_ids"] if frozen else None,
        )
    return RunPlan(
        cells=tuple(cells),
        diagnostics=diagnostics,
        frozen_design=frozen,
        extension=extension_plan,
        personas=tuple(personas),
        prompts=tuple(prompts),
    )


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
        kind=args.probe,
        n_models=args.n_models,
    )

    out = args.out or str(
        RESULTS_DIR
        / spec.safe_dir_name()
        / f"{spec_probe.label().replace('/', '-')}.jsonl"
    )

    # -- the grid ----------------------------------------------------------
    if args.extend_from and str(Path(out).resolve()) in {
        str(Path(value).resolve()) for value in args.extend_from
    }:
        raise SystemExit("--extend-from and --out must be different files")

    try:
        plan = build_plan(args, spec_probe)
    except RuntimeError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    cells = list(plan.cells)
    personas, dilemmas, diag = plan.personas, plan.prompts, plan.diagnostics
    frozen_design, extension_plan = plan.frozen_design, plan.extension

    expected_manifest = build_manifest(
        args=args,
        spec=spec,
        probe=spec_probe,
        resolved_file=spec.quantization.resolved_file,
        extension=extension_plan.identity if extension_plan else None,
        frozen_design=frozen_design,
        coordinates=plan.coordinates,
    )
    if extension_plan:
        for source in extension_plan.sources:
            validate_compatible_run(source.manifest, expected_manifest)
        # Every prior wave, not a sample: the shards are the thing this run is
        # being joined to, so a prompt change anywhere in them matters.
        verify_prompt_digests(
            extension_plan.base_rows,
            personas,
            dilemmas,
            spec_probe,
            args.system,
            four_dims_prompt_version=args.four_dims_prompt_version,
        )

    output_path = Path(out)
    has_output = output_path.is_file() and output_path.stat().st_size > 0
    if not args.plan_only:
        if args.no_resume and has_output:
            print(
                f"error: --no-resume requires a new or empty --out: {out}",
                file=sys.stderr,
            )
            return 2
        if args.overwrite:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("", encoding="utf-8")
            manifest_path(out).unlink(missing_ok=True)
            has_output = False

    try:
        run_manifest, adopted = reconcile_manifest(
            out,
            expected_manifest,
            has_output=has_output,
            write=not args.plan_only,
        )
    except RuntimeError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    run_id = run_manifest["run_id"]

    print(
        f"model:   {spec.alias}  ({spec.backend}, {spec.quantization.label}, "
        f"T={spec.temperature}, top_p={spec.top_p}, max_tokens={spec.max_output_tokens})"
    )
    print(f"probe:   {spec_probe.label()}")
    print(f"run:     {run_id}")
    if adopted:
        print(
            f"         continuing the existing run; this configuration would "
            f"otherwise be {expected_manifest['run_id']}"
        )
    if frozen_design:
        print(f"design:  {frozen_design['design_id']} ({frozen_design['path']})")
    if extension_plan:
        sources = ", ".join(str(source.path) for source in extension_plan.sources)
        print(
            f"extends: {sources} "
            f"({len(extension_plan.base_coordinates)} cell(s) already collected)"
        )
        print(
            f"adds:    {len(extension_plan.added_persona_ids)} new people + "
            f"{len(extension_plan.added_prompt_ids)} new dilemmas; "
            f"full union={len(extension_plan.all_persona_ids)}x"
            f"{len(extension_plan.all_prompt_ids)} "
            f"= {len(extension_plan.target_coordinates)} cells"
        )
    print(grid.summarize(cells))
    unusable = int((~diag.usable).sum()) if diag is not None and len(diag) else 0
    recovered = int(diag.recovered.sum()) if diag is not None and len(diag) else 0
    if unusable:
        print(f"note:    {unusable} persona transcript(s) unusable and excluded")
    if recovered:
        print(
            f"note:    {recovered} transcript(s) needed salvage parsing -- "
            "flagged per row as persona_recovered"
        )

    if has_output:
        try:
            check_resumable(out, cells, run_id, personas, dilemmas, spec_probe, args)
        except RuntimeError as err:
            print(f"error: {err}", file=sys.stderr)
            return 2

    done = set() if args.no_resume else completed_keys(out)
    todo = [
        c
        for c in cells
        if grid.cell_key(spec.alias, spec_probe.label(), c, run_id) not in done
    ]
    if done:
        print(
            f"resume:  {len(cells) - len(todo)} of {len(cells)} cells already in {out}"
        )
    if args.limit:
        todo = todo[: args.limit]
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
        print(
            f"  {i}/{len(todo)} | {rate:.2f} cells/s | ETA {fmt_duration(eta)} "
            f"| errors={errors}",
            flush=True,
        )

    if spec.quantization.format == "gguf" and not args.dry_run:
        print(f"quant file: {spec.quantization.resolved_file}")

    with single_writer(out), build_adapter(spec, dry_run=args.dry_run) as adapter:
        plan = adapter.thinking_plan(want_thinking=args.thinking)
        print(
            f"thinking: {plan.applied}"
            f"{'' if plan.standardized else '  (NOT standardized -- recorded per row)'}"
        )

        def run_one(cell):
            """-> (record). Never raises: a failed cell is a row with an error,
            so the run continues and that cell is retried next time."""
            key = grid.cell_key(spec.alias, spec_probe.label(), cell, run_id)
            conv = build_conversation(
                cell,
                spec_probe,
                args.system,
                four_dims_prompt_version=args.four_dims_prompt_version,
            )
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
            ordered = sorted(
                todo, key=lambda c: (len(c.prompt.text), c.persona.n_turns)
            )
            for start in range(0, len(ordered), spec.batch_size):
                chunk = ordered[start : start + spec.batch_size]
                keys = [
                    grid.cell_key(spec.alias, spec_probe.label(), c, run_id)
                    for c in chunk
                ]
                convs = [
                    build_conversation(
                        c,
                        spec_probe,
                        args.system,
                        four_dims_prompt_version=args.four_dims_prompt_version,
                    )
                    for c in chunk
                ]
                try:
                    raws = adapter.chat_batch(convs, plan=plan)
                    err = ""
                except Exception as e:
                    raws = [""] * len(chunk)
                    err = f"{type(e).__name__}: {e}"
                    print(f"  [batch error] {err}", flush=True)
                for cell, key, conv, raw in zip(chunk, keys, convs, raws):
                    buffer.append(
                        make_record(
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
                        )
                    )
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
    print(
        f"\nwrote {written} row(s) in {fmt_duration(elapsed)} "
        f"({errors} error(s)) -> {out}"
    )
    if killer.stop:
        print("stopped early -- re-run the same command to continue.")
    print(f"next: python -m syco parse {out}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
