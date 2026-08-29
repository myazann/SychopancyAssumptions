"""Experiment profiles shared by plan, run-all, status, and merge."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from syco import paths
from syco.model_registry import ModelRegistry, ModelSpec, load_registry
from syco.prompts import PROBE_KINDS, ProbeSpec

EXPERIMENTS_DIR = paths.CONFIG_DIR / "experiments"


def _positive(name: str, value, *, allow_none: bool = False):
    if value is None and allow_none:
        return
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def _range(name: str, value, *, minimum: float, maximum: float | None = None):
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric, got {value!r}")
    if value < minimum or (maximum is not None and value > maximum):
        bound = (
            f"between {minimum} and {maximum}"
            if maximum is not None
            else f">= {minimum}"
        )
        raise ValueError(f"{name} must be {bound}, got {value!r}")


def _resolve_path(value: str | None, default: Path) -> Path:
    if value is None:
        return default
    path = Path(value)
    return path if path.is_absolute() else paths.ROOT / path


@dataclass(frozen=True)
class ExperimentProfile:
    name: str
    path: Path
    model_selection: str | tuple[str, ...]
    probe: dict
    design: dict
    execution: dict
    output: dict

    @property
    def probe_spec(self) -> ProbeSpec:
        return ProbeSpec(
            kind=self.probe.get("kind", "openended"),
            n_models=self.probe.get("n_models", 3),
        )

    @property
    def results_dir(self) -> Path:
        return _resolve_path(self.output.get("results_dir"), paths.RESULTS_DIR)

    @property
    def logs_dir(self) -> Path:
        return _resolve_path(self.output.get("logs_dir"), self.results_dir)

    @property
    def merge_dir(self) -> Path:
        return _resolve_path(self.output.get("merge_dir"), self.results_dir / "merged")

    @property
    def collection_dir(self) -> Path | None:
        value = self.output.get("collection_dir")
        return _resolve_path(value, paths.RESULTS_DIR) if value else None

    def select_models(self, registry: ModelRegistry) -> list[ModelSpec]:
        if self.model_selection == "enabled":
            models = registry.select()
        else:
            models = [registry.get(alias) for alias in self.model_selection]
        if not models:
            raise ValueError(f"Experiment profile {self.name!r} selects no models")
        return models

    def output_for(self, spec: ModelSpec) -> Path:
        tag = self.probe_spec.label().replace("/", "-")
        return self.results_dir / spec.safe_dir_name() / f"{tag}.jsonl"

    def collection_output_for(self, spec: ModelSpec) -> Path | None:
        if self.collection_dir is None:
            return None
        tag = self.probe_spec.label().replace("/", "-")
        return self.collection_dir / spec.safe_dir_name() / f"{tag}.jsonl"

    def analysis_output_for(self, spec: ModelSpec) -> Path:
        return self.collection_output_for(spec) or self.output_for(spec)

    @property
    def is_extension(self) -> bool:
        return bool(self.design.get("extend_from"))

    @property
    def design_path(self) -> Path | None:
        value = self.design.get("lock")
        return _resolve_path(value, paths.ROOT) if value else None

    def extension_bases_for(self, spec: ModelSpec) -> list[Path]:
        """Every shard already collected for this model, oldest wave first.

        `design.extend_from` is one template or a list of them, so a third wave
        names the first two and the runner works out what is still missing.
        """
        template = self.design.get("extend_from")
        if not template:
            return []
        templates = [template] if isinstance(template, str) else list(template)
        return [
            _resolve_path(
                str(value).format(
                    model=spec.safe_dir_name(),
                    probe=self.probe_spec.label().replace("/", "-"),
                ),
                paths.RESULTS_DIR,
            )
            for value in templates
        ]

    def extension_base_for(self, spec: ModelSpec) -> Path | None:
        bases = self.extension_bases_for(spec)
        return bases[0] if bases else None

    def log_for(self, spec: ModelSpec) -> Path:
        tag = self.probe_spec.label().replace("/", "-")
        return self.logs_dir / spec.safe_dir_name() / f"{tag}.log"

    def merged_output(self) -> Path:
        tag = self.probe_spec.label().replace("/", "-")
        return self.merge_dir / f"all_{tag}.jsonl"

    def parsed_output_for(self, spec: ModelSpec, *, format: str = "parquet") -> Path:
        raw = self.analysis_output_for(spec)
        stem = str(raw).removesuffix(".jsonl")
        return Path(f"{stem}{self.probe_spec.parsed_table_suffix}.{format}")

    def run_args(self, spec: ModelSpec) -> list[str]:
        args = [
            "--model",
            spec.alias,
            "--out",
            str(self.output_for(spec)),
            "--probe",
            self.probe_spec.kind,
            "--system",
            str(self.probe.get("system") or ""),
            "--n-reps",
            str(self.design.get("n_reps", 1)),
            "--seed",
            str(self.design.get("seed", 1000)),
        ]
        if self.probe_spec.family == "open-ended":
            args.extend(("--n-models", str(self.probe_spec.n_models)))
        for extension_base in self.extension_bases_for(spec):
            args.extend(("--extend-from", str(extension_base)))
        if self.design_path is not None:
            args.extend(("--design", str(self.design_path)))
        for key, flag in (("n_personas", "--n-personas"), ("n_prompts", "--n-prompts")):
            value = self.design.get(key)
            if value is not None:
                args.extend((flag, str(value)))
        for key, flag in (
            ("persona_types", "--persona-types"),
            ("prompt_types", "--prompt-types"),
        ):
            values = self.design.get(key)
            if values:
                args.append(flag)
                args.extend(str(value) for value in values)
        if not self.design.get("include_control", True):
            args.append("--no-control")
        if self.probe.get("thinking", False):
            args.append("--thinking")
        generation = self.probe.get("generation") or {}
        for key, flag in (
            ("temperature", "--temperature"),
            ("top_p", "--top-p"),
            ("max_tokens", "--max-tokens"),
            ("batch_size", "--batch-size"),
            ("max_workers", "--max-workers"),
        ):
            if generation.get(key) is not None:
                args.extend((flag, str(generation[key])))
        return args

    def build_cells(self, spec: ModelSpec | None = None):
        """The exact cells this profile administers for `spec`.

        Delegates to the runner's own planner rather than reimplementing it.
        A second copy of this logic is how `syco status` drifted away from what
        `syco run` actually did: the runner resolved `--design` into its
        arguments first, the status path did not, and every extension profile
        reported as unplannable.
        """
        from scripts.run_assumptions import build_plan, parse_args

        if spec is None:
            specs = self.select_models(load_registry())
            if not specs:
                raise ValueError(f"profile {self.name} selects no models")
            spec = specs[0]
        args = parse_args(self.run_args(spec))
        plan = build_plan(args, self.probe_spec)
        return list(plan.cells), plan.diagnostics


def profile_path(name_or_path: str = "default") -> Path:
    candidate = Path(name_or_path)
    if candidate.is_file():
        return candidate.resolve()
    if candidate.suffix or len(candidate.parts) > 1:
        candidate = candidate if candidate.is_absolute() else paths.ROOT / candidate
        if not candidate.is_file():
            raise FileNotFoundError(f"Experiment profile not found: {candidate}")
        return candidate.resolve()
    path = EXPERIMENTS_DIR / f"{name_or_path}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Unknown experiment profile {name_or_path!r}: {path}")
    return path


def load_profile(name_or_path: str = "default") -> ExperimentProfile:
    path = profile_path(name_or_path)
    with path.open(encoding="utf-8") as handle:
        doc = yaml.safe_load(handle) or {}
    models = doc.get("models", "enabled")
    if models != "enabled":
        if not isinstance(models, list) or not models:
            raise ValueError("profile `models` must be `enabled` or a non-empty list")
        models = tuple(str(model) for model in models)
    probe = dict(doc.get("probe") or {})
    design = dict(doc.get("design") or {})
    execution = dict(doc.get("execution") or {})
    output = dict(doc.get("output") or {})

    if probe.get("kind", "openended") not in set(PROBE_KINDS):
        raise ValueError("profile probe.kind must be one of: " + ", ".join(PROBE_KINDS))
    if probe.get("kind", "openended") == "openended":
        _positive("probe.n_models", probe.get("n_models", 3))
    _positive("design.n_personas", design.get("n_personas"), allow_none=True)
    _positive("design.n_prompts", design.get("n_prompts"), allow_none=True)
    _positive("design.n_reps", design.get("n_reps", 1))
    extend_from = design.get("extend_from")
    if extend_from is not None and not (
        isinstance(extend_from, str)
        or (
            isinstance(extend_from, list)
            and all(isinstance(value, str) for value in extend_from)
        )
    ):
        raise ValueError(
            "design.extend_from must be a path template string, or a list of them"
        )
    if design.get("lock") is not None and not isinstance(design.get("lock"), str):
        raise ValueError("design.lock must be a path string")
    _positive("execution.poll_seconds", execution.get("poll_seconds", 2))
    _positive("execution.wait_report_seconds", execution.get("wait_report_seconds", 60))
    _positive(
        "execution.wait_timeout_seconds",
        execution.get("wait_timeout_seconds"),
        allow_none=True,
    )
    generation = probe.get("generation") or {}
    _range("probe.generation.temperature", generation.get("temperature"), minimum=0)
    _range(
        "probe.generation.top_p",
        generation.get("top_p"),
        minimum=0,
        maximum=1,
    )
    for key in ("max_tokens", "batch_size", "max_workers"):
        _positive(
            f"probe.generation.{key}",
            generation.get(key),
            allow_none=True,
        )
    return ExperimentProfile(
        name=str(doc.get("name") or path.stem),
        path=path,
        model_selection=models,
        probe=probe,
        design=design,
        execution=execution,
        output=output,
    )
