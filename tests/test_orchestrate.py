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
    profile = FakeProfile(tmp_path, specs,
                          model_selection=("small", "large", "medium"))
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
