from pathlib import Path

from syco.experiments import ExperimentProfile
from syco.model_registry import ModelSpec
from syco.orchestrate import GPU, run_all


class FakeProfile:
    name = "test"
    execution = {"poll_seconds": 1, "wait_report_seconds": 1}
    model_selection = "enabled"

    def __init__(self, root, specs, model_selection="enabled"):
        self.model_selection = model_selection
        self.path = root / "test.yaml"
        self.results_dir = root / "results"
        self.logs_dir = root / "logs"
        self.specs = specs

    def select_models(self, registry):
        return self.specs

    def run_args(self, spec):
        return ["--model", spec.alias, "--out", str(self.output_for(spec))]

    def output_for(self, spec):
        return self.results_dir / f"{spec.alias}.jsonl"

    def log_for(self, spec):
        return self.logs_dir / f"{spec.alias}.log"


def _spec(alias, vram):
    return ModelSpec(
        alias=alias,
        family="gemma",
        ref=f"org/{alias}-GGUF",
        backend="llamacpp",
        estimated_vram_mib=vram,
    )


def test_run_all_schedules_largest_first_and_forwards_limit(tmp_path, monkeypatch):
    specs = [_spec("small", 4_000), _spec("large", 9_000), _spec("medium", 6_000)]
    profile = FakeProfile(tmp_path, specs)
    launches = []

    class Process:
        def __init__(self, command, **kwargs):
            launches.append((command, kwargs["env"]["CUDA_VISIBLE_DEVICES"]))

        def poll(self):
            return 0

    monkeypatch.setattr("syco.orchestrate.load_registry", lambda: object())
    monkeypatch.setattr(
        "syco.orchestrate.query_gpus",
        lambda: [GPU(0, 12_000, 12_000), GPU(1, 12_000, 12_000)],
    )
    monkeypatch.setattr("syco.orchestrate.subprocess.Popen", Process)
    monkeypatch.setattr("syco.orchestrate.time.sleep", lambda _: None)

    assert run_all(profile, limit_per_model=7) == 0
    aliases = [command[command.index("--model") + 1] for command, _ in launches]
    assert aliases[:2] == ["large", "medium"]
    assert set(aliases) == {"large", "medium", "small"}
    assert all(command[-2:] == ["--limit", "7"] for command, _ in launches)
    assert {gpu for _, gpu in launches[:2]} == {"0", "1"}
    assert all(Path(profile.log_for(spec)).is_file() for spec in specs)


def test_explicit_model_list_is_run_in_the_order_written(tmp_path, monkeypatch):
    """A `models:` list in the profile is a running order, not just a filter --
    largest-first is only the fallback for `models: enabled`."""
    specs = [_spec("small", 4_000), _spec("large", 9_000), _spec("medium", 6_000)]
    profile = FakeProfile(tmp_path, specs, model_selection=("small", "large", "medium"))
    launches = []

    class Process:
        def __init__(self, command, **kwargs):
            launches.append(command[command.index("--model") + 1])

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr("syco.orchestrate.load_registry", lambda: object())
    monkeypatch.setattr(
        "syco.orchestrate.query_gpus",
        lambda: [GPU(index=0, total_mib=24_000, free_mib=24_000)],
    )
    monkeypatch.setattr("syco.orchestrate.subprocess.Popen", Process)
    monkeypatch.setattr("syco.orchestrate.time.sleep", lambda _: None)

    run_all(profile)
    assert launches == ["small", "large", "medium"]


def test_structured_profile_uses_probe_specific_paths_and_arguments(tmp_path):
    spec = _spec("model", 4_000)
    profile = ExperimentProfile(
        name="structured",
        path=tmp_path / "structured.yaml",
        model_selection=(spec.alias,),
        probe={"kind": "4dims", "n_models": 99},
        design={"n_reps": 1, "seed": 1000},
        execution={},
        output={"results_dir": str(tmp_path / "results")},
    )

    assert profile.output_for(spec).name == "4dims.jsonl"
    assert profile.parsed_output_for(spec).name == "4dims_structured.parquet"
    args = profile.run_args(spec)
    assert args[args.index("--probe") + 1] == "4dims"
    assert "--n-models" not in args


def test_extension_profile_resolves_model_base_and_separate_collection(tmp_path):
    spec = _spec("model", 4_000)
    profile = ExperimentProfile(
        name="extension",
        path=tmp_path / "extension.yaml",
        model_selection=(spec.alias,),
        probe={"kind": "4dims"},
        design={
            "lock": "config/designs/structured.json",
            "n_personas": 20,
            "n_prompts": 20,
            "n_reps": 1,
            "seed": 1000,
            "extend_from": str(tmp_path / "base" / "{model}" / "{probe}.jsonl"),
        },
        execution={},
        output={
            "results_dir": str(tmp_path / "extensions"),
            "collection_dir": str(tmp_path / "collections"),
        },
    )

    args = profile.run_args(spec)
    assert args[args.index("--extend-from") + 1].endswith("base/model/4dims.jsonl")
    assert args[args.index("--design") + 1].endswith(
        "config/designs/structured.json"
    )
    assert profile.output_for(spec) != profile.extension_base_for(spec)
    assert profile.analysis_output_for(spec) == profile.collection_output_for(spec)
    assert profile.parsed_output_for(spec).parent == tmp_path / "collections" / "model"


def test_orchestration_plans_exactly_what_the_runner_would_run(tmp_path, monkeypatch):
    """`status` and `merge` must not re-derive the grid their own way.

    They used to: the runner resolved `--design` into its arguments before
    planning and orchestration did not, so every extension profile reported as
    unplannable while the runs themselves were fine. Both now go through
    `build_plan`, and this pins that they agree.
    """
    from scripts.run_assumptions import build_plan, parse_args
    from syco.orchestrate import _planned_run

    spec = _spec("model", 4_000)
    profile = ExperimentProfile(
        name="structured",
        path=tmp_path / "structured.yaml",
        model_selection=(spec.alias,),
        probe={"kind": "4dims"},
        design={"n_personas": 2, "n_prompts": 2, "n_reps": 1, "seed": 1000},
        execution={},
        output={"results_dir": str(tmp_path / "results")},
    )

    class Registry:
        def get(self, alias):
            return spec

        def with_resolved_quant(self, value):
            return value

    from syco.prompts import ProbeSpec

    args = parse_args(profile.run_args(spec))
    runner_plan = build_plan(args, ProbeSpec(kind="4dims", n_models=3))
    orchestrated, manifest = _planned_run(profile, Registry(), spec)

    assert set(orchestrated.coordinates) == set(runner_plan.coordinates)
    assert manifest["identity"]["design"]["cells"] == len(runner_plan.cells)
    assert profile.build_cells(spec)[0] == list(runner_plan.cells)


def test_several_prior_waves_are_all_passed_to_the_runner(tmp_path):
    spec = _spec("model", 4_000)
    profile = ExperimentProfile(
        name="wave3",
        path=tmp_path / "wave3.yaml",
        model_selection=(spec.alias,),
        probe={"kind": "4dims"},
        design={
            "lock": "config/designs/structured-60x60.json",
            "n_reps": 1,
            "seed": 1000,
            "extend_from": [
                str(tmp_path / "base" / "{model}" / "{probe}.jsonl"),
                str(tmp_path / "wave2" / "{model}" / "{probe}.jsonl"),
            ],
        },
        execution={},
        output={
            "results_dir": str(tmp_path / "wave3"),
            "collection_dir": str(tmp_path / "collections"),
        },
    )

    args = profile.run_args(spec)
    bases = [args[i + 1] for i, value in enumerate(args) if value == "--extend-from"]

    assert len(bases) == 2
    assert bases[0].endswith("base/model/4dims.jsonl")
    assert bases[1].endswith("wave2/model/4dims.jsonl")
    assert len(profile.extension_bases_for(spec)) == 2


def test_models_merge_despite_per_model_sources_and_schema_drift():
    """`merge` compares the design across models, not which files it read.

    Each model's wave is built on that model's own earlier shards, so the paths
    and run IDs differ by construction, and a study collected over months spans
    two manifest schema versions. Comparing whole identity dicts refused both.
    """
    from syco.manifest import identity_conflicts

    legacy = {
        "identity": {
            "model": {"alias": "a"},
            "instrument": {"probe": "4dims", "system": "", "thinking": False},
            "data": {"personas_sha256": "p", "prompts_sha256": "q"},
            "design": {
                "persona_types": None,
                "prompt_types": ["original_post"],
                "n_reps": 1,
                "include_control": True,
                "seed": 1000,
            },
        }
    }
    current = {
        "identity": {
            "model": {"alias": "b"},
            "instrument": {"probe": "4dims", "system": "", "thinking": False},
            "data": {"personas_sha256": "p", "prompts_sha256": "q"},
            "design": {
                "persona_types": None,
                "prompt_types": ["original_post"],
                "n_reps": 1,
                "include_control": True,
                "seed": None,
                "coordinates_sha256": "abc",
                "frozen_design_id": "lock",
            },
        }
    }
    sections = ("instrument", "data", "design")

    assert identity_conflicts(legacy, current, sections=sections) == []

    current["identity"]["instrument"]["probe"] = "supporttypes"
    assert identity_conflicts(legacy, current, sections=sections)
