from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from syco.linear_probe.activations import BlockCollector, pool_hidden
from syco.linear_probe.artifacts import (
    append_jsonl,
    balanced_sparse_pairs,
    paths_for,
    read_jsonl,
)
from syco.linear_probe.cli import _in_memory_plan
from syco.linear_probe.config import (
    ProbePipelineConfig,
    SplitConfig,
    load_config,
    validate,
)
from syco.linear_probe.dataset import assign_splits
from syco.linear_probe.evaluation import _effect_record, _pair
from syco.linear_probe.labels import (
    InvalidLabel,
    parse_labels,
    raw_label_paths,
    run_labeling,
    strict_parse_label,
)
from syco.linear_probe.modeling import candidate_block_indices, resolve_decoder_blocks
from syco.linear_probe.prompts import build_label_prompt
from syco.linear_probe.steering import SteeringHook, score_options
from syco.linear_probe.training import (
    absolute_extreme_auc,
    fit_ridge_raw,
    regression_metrics,
)
from syco.prompts import STRUCTURED_CONTAINER, STRUCTURED_DIMENSIONS


def _valid_label(kind="4dims"):
    container = STRUCTURED_CONTAINER[kind]
    return json.dumps({"mental_model": {container: {
        dimension: {"score": 0.2 + index / 10, "explanation": "audit"}
        for index, dimension in enumerate(STRUCTURED_DIMENSIONS[kind])
    }}})


def test_default_config_is_strict_and_leakage_safe():
    config = load_config("config/linear_probe.yaml")
    assert [(model.id, model.model) for model in config.labeling.models] == [
        ("qwen36_35b_a3b", "Qwen3.6-35B-A3B"),
        ("gemma3_27b", "Gemma3-27B"),
    ]
    assert config.labeling.aggregation == "mean"
    assert config.training.split.group_by == "two_axis"
    assert config.target.overlength == "error"
    assert config.target.pooling == "final_user_mean"
    assert config.target.hf_ref is None
    assert not config.target.layers.explicit
    assert config.target.layers.fractions[-1] == 1.0
    assert all(model.model_revision for model in config.labeling.models)
    assert all(model.model_sha256 for model in config.labeling.models)
    assert config.steering.answer_yes == "Yes"
    assert config.steering.answer_no == "No"
    with pytest.raises(TypeError, match="must be strings"):
        validate(replace(
            config,
            steering=replace(config.steering, answer_yes=True),
        ))


def test_stage_digests_reuse_upstream_artifacts_for_larger_alpha_grid():
    config = load_config("config/linear_probe.yaml")
    expanded = replace(
        config,
        steering=replace(config.steering, alphas=(-2.0, -1.0, 0.0, 1.0, 2.0)),
    )
    for stage in ("dataset", "labels", "activations", "probes"):
        assert config.stage_digest(stage) == expanded.stage_digest(stage)
    assert config.stage_digest("steering") != expanded.stage_digest("steering")
    assert config.stage_digest("evaluation") != expanded.stage_digest("evaluation")
    first, second = paths_for(config), paths_for(expanded)
    assert first.labels == second.labels
    assert first.probes == second.probes
    assert first.steering != second.steering


def test_selecting_target_reuses_labels_but_changes_activations():
    config = load_config("config/linear_probe.yaml")
    selected = replace(
        config,
        target=replace(
            config.target,
            model="example-target",
            hf_ref="organization/example-target",
            tokenizer_ref="organization/example-target",
            revision="0" * 40,
        ),
    )
    assert config.stage_digest("dataset") == selected.stage_digest("dataset")
    assert config.stage_digest("labels") == selected.stage_digest("labels")
    assert config.stage_digest("activations") != selected.stage_digest("activations")


def test_sparse_pairs_are_balanced_and_repeatable():
    people = [f"p{i}" for i in range(7)]
    prompts = [f"q{i}" for i in range(11)]
    first = balanced_sparse_pairs(people, prompts, 4, 9)
    second = balanced_sparse_pairs(people, prompts, 4, 9)
    assert first == second
    assert len(first) == 28
    assert {sum(person == p for person, _ in first) for p in people} == {4}
    prompt_degrees = [sum(prompt == q for _, prompt in first) for q in prompts]
    assert max(prompt_degrees) - min(prompt_degrees) <= 1


def test_two_axis_split_keeps_both_axes_disjoint():
    rows = pd.DataFrame([
        {"persona_id": person, "prompt_id": prompt,
         "design_unit_id": f"{person}-{prompt}", "prompt_type": framing}
        for person in [f"p{i}" for i in range(12)]
        for prompt in [f"q{i}" for i in range(12)]
        for framing in ("original_post", "flipped_story")
    ])
    split = assign_splits(rows, SplitConfig(group_by="two_axis", seed=4))
    primary = {name: split[split.split == name] for name in
               ("train", "validation", "test")}
    for left, right in (("train", "validation"), ("train", "test"),
                        ("validation", "test")):
        assert set(primary[left].persona_id).isdisjoint(primary[right].persona_id)
        assert set(primary[left].prompt_id).isdisjoint(primary[right].prompt_id)
    assert split.groupby(["persona_id", "prompt_id"]).split.nunique().max() == 1


def test_default_sparse_design_uses_only_primary_disjoint_pairs():
    config = load_config("config/linear_probe.yaml")
    table = _in_memory_plan(config)
    assert len(table) == 17_040
    assert table.split.value_counts().to_dict() == {
        "train": 11_928,
        "validation": 2_556,
        "test": 2_556,
    }
    assert table.groupby("design_unit_id").prompt_type.nunique().eq(2).all()
    real = table[table.persona_id != "none"]
    partitions = {
        name: real[real.split == name]
        for name in ("train", "validation", "test")
    }
    for left, right in (("train", "validation"), ("train", "test"),
                        ("validation", "test")):
        assert set(partitions[left].persona_id).isdisjoint(
            partitions[right].persona_id
        )
        assert set(partitions[left].prompt_id).isdisjoint(
            partitions[right].prompt_id
        )
    controls = table[table.persona_id == "none"]
    assert controls.split.value_counts().to_dict() == {
        "train": 168, "validation": 36, "test": 36,
    }


def test_label_prompt_is_paper_schema_without_advice_response():
    text = build_label_prompt("4dims", [], "Was I wrong?")
    assert '"validation_seeking"' in text
    assert "RESPONSE:" not in text
    assert text.rstrip().endswith("}")


@pytest.mark.parametrize("kind", ["4dims", "supporttypes"])
def test_strict_label_parser_accepts_only_exact_schema(kind):
    rows = strict_parse_label(_valid_label(kind), kind)
    assert [row["dimension"] for row in rows] == list(STRUCTURED_DIMENSIONS[kind])
    payload = json.loads(_valid_label(kind))
    container = STRUCTURED_CONTAINER[kind]
    first = STRUCTURED_DIMENSIONS[kind][0]

    corruptions = []
    extra = json.loads(_valid_label(kind))
    extra["extra"] = 1
    corruptions.append(extra)
    missing = json.loads(_valid_label(kind))
    del missing["mental_model"][container][first]
    corruptions.append(missing)
    string_score = json.loads(_valid_label(kind))
    string_score["mental_model"][container][first]["score"] = "0.7"
    corruptions.append(string_score)
    off_scale = json.loads(_valid_label(kind))
    off_scale["mental_model"][container][first]["score"] = 7
    corruptions.append(off_scale)
    nan_score = json.loads(_valid_label(kind))
    nan_score["mental_model"][container][first]["score"] = float("nan")
    corruptions.append(nan_score)
    for payload in corruptions:
        with pytest.raises(InvalidLabel):
            strict_parse_label(json.dumps(payload), kind)
    with pytest.raises(InvalidLabel):
        strict_parse_label("```json\n" + _valid_label(kind) + "\n```", kind)


def test_strict_label_parser_rejects_duplicate_keys_and_empty_explanations():
    duplicate = _valid_label("4dims").replace(
        '"mental_model":', '"mental_model": {}, "mental_model":', 1
    )
    with pytest.raises(InvalidLabel, match="duplicate JSON key"):
        strict_parse_label(duplicate, "4dims")
    payload = json.loads(_valid_label("4dims"))
    payload["mental_model"]["beliefs"]["validation_seeking"]["explanation"] = " "
    with pytest.raises(InvalidLabel, match="non-empty"):
        strict_parse_label(json.dumps(payload), "4dims")


def test_jsonl_resume_quarantines_only_a_truncated_tail(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_bytes(b'{"ok": 1}\n{"partial":')
    assert read_jsonl(path) == [{"ok": 1}]
    append_jsonl(path, [{"ok": 2}])
    assert read_jsonl(path) == [{"ok": 1}, {"ok": 2}]
    assert path.with_name("rows.jsonl.truncated").read_bytes() == b'{"partial":\n'


def test_dry_label_shards_are_disjoint_complete_and_resumable(tmp_path):
    base = load_config("config/linear_probe.yaml")
    config = replace(
        base,
        output_dir=str(tmp_path / "probe"),
        design=replace(
            base.design,
            n_persona_ids=3,
            n_prompt_ids=3,
            dilemmas_per_persona=1,
            persona_types=("assumptions",),
            include_control=False,
        ),
        labeling=replace(base.labeling, instruments=("4dims",)),
        steering=replace(
            base.steering,
            dimensions=tuple(STRUCTURED_DIMENSIONS["4dims"]),
        ),
    )
    validate(config)
    artifacts = paths_for(config, dry_run=True)
    with pytest.raises(ValueError, match="--teacher"):
        run_labeling(config, artifacts, dry_run=True, limit=1)
    runs = [
        run_labeling(
            config, artifacts, dry_run=True, num_shards=2,
            shard_index=shard_index, teacher_id=teacher.id,
        )
        for teacher in config.labeling.models
        for shard_index in range(2)
    ]
    assert sum(run["valid"] for run in runs) == 12
    paths = raw_label_paths(artifacts)
    key_sets = [{row["label_key"] for row in read_jsonl(path)} for path in paths]
    assert len(set().union(*key_sets)) == sum(map(len, key_sets))
    resumed = run_labeling(
        config, artifacts, dry_run=True, num_shards=2, shard_index=0,
        teacher_id=config.labeling.models[0].id,
    )
    assert resumed["planned"] == 0
    assert resumed["raw_path"]
    labels, quality = parse_labels(config, artifacts)
    assert quality["valid_completions"] == quality["expected_completions"] == 12
    assert set(quality["by_teacher"]) == {model.id for model in config.labeling.models}
    assert quality["teacher_agreement"]
    assert len(labels) == 12 * 4


def test_pooling_is_padding_safe():
    torch = pytest.importorskip("torch")
    hidden = torch.tensor([
        [[99., 99.], [1., 2.], [3., 4.]],
        [[5., 6.], [7., 8.], [99., 99.]],
    ])
    mask = torch.tensor([[0, 1, 1], [1, 1, 0]])
    mean = pool_hidden(hidden, mask, "attention_mean")
    last = pool_hidden(hidden, mask, "last_nonpadding")
    assert torch.equal(mean, torch.tensor([[2., 3.], [6., 7.]]))
    assert torch.equal(last, torch.tensor([[3., 4.], [7., 8.]]))


def test_block_resolver_and_collector_use_same_modules():
    torch = pytest.importorskip("torch")

    class Add(torch.nn.Module):
        def __init__(self, value):
            super().__init__()
            self.value = value

        def forward(self, hidden):
            return hidden + self.value

    blocks = torch.nn.ModuleList([Add(1), Add(2), Add(3)])

    class Fake(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = blocks

        def forward(self, hidden):
            for block in self.model.layers:
                hidden = block(hidden)
            return hidden

    model = Fake()
    resolved, path = resolve_decoder_blocks(model)
    assert resolved is blocks and path == "model.layers"

    gemma = torch.nn.Module()
    gemma.model = torch.nn.Module()
    gemma.model.language_model = torch.nn.Module()
    gemma.model.language_model.layers = blocks
    resolved, path = resolve_decoder_blocks(gemma)
    assert resolved is blocks and path == "model.language_model.layers"

    assert candidate_block_indices(3, SimpleNamespace(explicit=(0, 2), fractions=())) == (0, 2)
    collector = BlockCollector(blocks, (0, 2), "attention_mean")
    try:
        collector.begin(torch.ones((1, 2), dtype=torch.bool))
        model(torch.zeros((1, 2, 4)))
        values = collector.stacked((0, 2))
        assert values.shape == (1, 2, 4)
        assert torch.all(values[:, 0] == 1)
        assert torch.all(values[:, 1] == 6)
    finally:
        collector.close()


def test_ridge_raw_coefficients_and_absolute_auc():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(400, 5)).astype(np.float32)
    truth = np.array([.5, -.25, .1, 0, .3], dtype=np.float32)
    y = X @ truth + .2
    coefficient, intercept = fit_ridge_raw(X, y, alpha=1e-6, standardize=True)
    assert np.allclose(coefficient, truth, atol=2e-3)
    assert intercept == pytest.approx(.2, abs=2e-3)
    labels = np.array([.1, .2, .3, .7, .8, .9])
    prediction = np.arange(6)
    auc, low, high = absolute_extreme_auc(labels, prediction, .3, .7)
    assert auc == 1 and low == 2 and high == 2
    assert regression_metrics(labels, prediction)["auc_absolute"] == 1


def test_steering_hook_zero_identity_and_positive_sign():
    torch = pytest.importorskip("torch")

    class TupleLayer(torch.nn.Module):
        def forward(self, hidden):
            return hidden, "cache"

    layer = TupleLayer()
    source = torch.zeros((1, 2, 3))
    with SteeringHook(layer, np.array([1., 0., 0.])) as hook:
        zero = layer(source)
        assert zero[0] is source and zero[1] == "cache"
        hook.strength = 2.0
        moved = layer(source)
        assert torch.all(moved[0][..., 0] == 2)
        assert torch.all(moved[0][..., 1:] == 0)
        assert moved[1] == "cache"


def test_option_scorer_uses_conditional_log_likelihood():
    torch = pytest.importorskip("torch")

    class Tokenizer:
        pad_token_id = 0
        padding_side = "left"

        def encode(self, text, add_special_tokens=False):
            values = {"p": [1, 2], "long": [1, 2, 2], "Yes": [3], "No": [4]}
            for prompt in ("long", "p"):
                if text.startswith(prompt) and text != prompt:
                    return values[prompt] + values[text[len(prompt):]]
            return values[text]

        def pad(self, features, padding=True, return_tensors="pt"):
            width = max(len(item["input_ids"]) for item in features)
            ids, mask = [], []
            for item in features:
                values = item["input_ids"]
                n = width - len(values)
                ids.append([0] * n + values)
                mask.append([0] * n + [1] * len(values))
            return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)}

    class Model(torch.nn.Module):
        def forward(self, input_ids, attention_mask, **kwargs):
            logits = torch.zeros((*input_ids.shape, 6))
            logits[..., 3] = 1.0
            logits[..., 4] = 2.0
            return SimpleNamespace(logits=logits)

    scores = score_options(
        Model(), Tokenizer(), ["p", "long"], "Yes", "No", torch.device("cpu"),
        max_length=20, batch_size=2,
    )
    assert scores[0]["logit_no"] == pytest.approx(1.0)
    assert scores[1]["logit_no"] == pytest.approx(1.0)
    assert scores[0]["p_no"] == pytest.approx(1 / (1 + np.exp(-1)))

    tokenizer = Tokenizer()
    tokenizer.padding_side = "right"
    with pytest.raises(ValueError, match="padding_side"):
        score_options(
            Model(), tokenizer, ["p"], "Yes", "No", torch.device("cpu"),
            max_length=20, batch_size=1,
        )


def test_evaluation_never_adds_post_treatment_eligible_cells():
    config = ProbePipelineConfig()
    baseline = pd.DataFrame({
        "design_unit_id": ["a", "b"], "persona_type": ["x", "x"],
        "persona_id": ["p1", "p2"], "prompt_id": ["q1", "q2"], "rep": [0, 0],
        "p_no_original_post": [.9, .4], "p_no_flipped_story": [.8, .2],
        "candidate_mass_original_post": [.9, .9],
        "candidate_mass_flipped_story": [.9, .9],
    })
    current = baseline.copy()
    current["p_no_original_post"] = [.8, .95]  # b becomes eligible only after treatment
    current["p_no_flipped_story"] = [.4, .99]
    record, rows = _effect_record(current, baseline, config, "user_rightness", "probe", -1)
    assert record["eligible_fixed_n"] == 1
    assert record["hard_delta"] == -1
    assert rows.set_index("design_unit_id").loc["b", "eligible_at_alpha_zero"] == 0


def test_evaluation_refuses_missing_framings_or_treatment_units():
    frame = pd.DataFrame({
        "design_unit_id": ["a"], "persona_type": ["x"],
        "persona_id": ["p"], "prompt_id": ["q"], "rep": [0],
        "prompt_type": ["original_post"], "logp_yes": [-1.0],
        "logp_no": [-2.0], "logit_no": [-1.0], "p_no": [.2],
        "log_candidate_mass": [-.7], "candidate_mass": [.5],
    })
    with pytest.raises(ValueError, match="same design units"):
        _pair(frame)

    config = ProbePipelineConfig()
    baseline = pd.DataFrame({
        "design_unit_id": ["a", "b"], "persona_type": ["x", "x"],
        "persona_id": ["p1", "p2"], "prompt_id": ["q1", "q2"],
        "rep": [0, 0], "p_no_original_post": [.9, .9],
        "p_no_flipped_story": [.8, .8],
        "candidate_mass_original_post": [.9, .9],
        "candidate_mass_flipped_story": [.9, .9],
    })
    with pytest.raises(ValueError, match="exact alpha-zero unit set"):
        _effect_record(
            baseline.iloc[:1], baseline, config, "user_rightness", "probe", 1,
        )
