from pathlib import Path

from syco.model_registry import ModelSpec
from syco.orchestrate import GPU, run_all


class FakeProfile:
    name = "test"
    execution = {"poll_seconds": 1, "wait_report_seconds": 1}

    def __init__(self, root, specs):
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
