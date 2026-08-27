import math

import pandas as pd
import pytest

from scripts.summarize_assumptions import (
    confidence,
    structured_framing_flip,
    structured_scores,
    top_labels,
)
from syco.data import NO_PERSONA, Persona, Prompt
from syco.grid import build_cells


def _persona(kind, person):
    return Persona(person, kind, (), False, 0)


def test_restrict_cells_limits_to_full_design_coordinates():
    """restrict_cells is keyed on all four coordinates, so restricting one
    (facet, person, framing, dilemma) never silently admits another framing."""
    personas = [_persona("age", "p1"), _persona("gender", "p1")]
    prompts = [
        Prompt("q1", "original_post", "original"),
        Prompt("q1", "flipped_story", "flipped"),
    ]
    keep = {("age", "p1", "original_post", "q1")}
    cells = build_cells(personas, prompts, include_no_persona=False,
                        restrict_cells=keep)
    got = {(c.persona.persona_type, c.persona.persona_id,
            c.prompt.prompt_type, c.prompt.prompt_id) for c in cells}
    assert got == keep



def test_partial_grid_interleaves_source_order_facets_and_control():
    personas = [
        _persona("hobbies", "p1"), _persona("politics", "p1"),
        _persona("hobbies", "p2"), _persona("politics", "p2"),
    ]
    prompts = [Prompt("q1", "original_post", "original")]

    cells = build_cells(personas, prompts, prompt_types=["original_post"])

    assert [(c.persona.persona_type, c.persona.persona_id) for c in cells] == [
        (NO_PERSONA, NO_PERSONA),
        ("hobbies", "p1"),
        ("politics", "p1"),
        ("hobbies", "p2"),
        ("politics", "p2"),
    ]


def _summary_rows(model, control_label, facet_label):
    rows = []
    for persona_type, persona_id, label in (
        (NO_PERSONA, NO_PERSONA, control_label),
        ("age", "p1", facet_label),
    ):
        for rank, probability in enumerate((0.6, 0.3, 0.1)):
            rows.append({
                "run_id": f"run-{model}",
                "probe": "openended3",
                "persona_type": persona_type,
                "persona_id": persona_id,
                "prompt_type": "original_post",
                "prompt_id": "q1",
                "rep": 0,
                "rank": rank,
                "label": label if rank == 0 else f"other-{rank}",
                "probability_norm": probability,
            })
    return rows


def test_summaries_never_pool_runs_or_control_baselines():
    df = pd.DataFrame(
        _summary_rows("model-a", "alpha", "alpha")
        + _summary_rows("model-b", "beta", "alpha")
    )

    conf = confidence(df).reset_index()
    assert len(conf) == 4
    assert set(conf.run_id) == {"run-model-a", "run-model-b"}
    assert conf.entropy_bits.max() <= math.log2(3)

    labels = top_labels(df, n=1)
    model_a = labels[(labels.run_id == "run-model-a") &
                     (labels.persona_type == "age")].iloc[0]
    model_b = labels[(labels.run_id == "run-model-b") &
                     (labels.persona_type == "age")].iloc[0]
    assert model_a["lift"] == 1.0
    assert math.isnan(model_b["lift"])


def test_structured_summary_uses_matching_control_and_paired_framings():
    rows = []
    for persona_type, persona_id, original, flipped in (
        (NO_PERSONA, NO_PERSONA, 0.2, 0.4),
        ("age", "p1", 0.7, 0.8),
    ):
        for framing, score in (("original_post", original),
                               ("flipped_story", flipped)):
            rows.append({
                "run_id": "run-structured", "probe": "4dims",
                "persona_type": persona_type, "persona_id": persona_id,
                "prompt_type": framing, "prompt_id": "q1", "rep": 0,
                "dimension": "validation_seeking", "score": score,
                "parse_status": "clean",
            })
    df = pd.DataFrame(rows)
    scores = structured_scores(df)
    age_original = scores[(scores.persona_type == "age") &
                          (scores.prompt_type == "original_post")].iloc[0]
    assert age_original.mean_score == 0.7
    assert age_original.control_mean == 0.2
    assert age_original.delta_vs_control == pytest.approx(0.5)

    flip = structured_framing_flip(df)
    age = flip[flip.persona_type == "age"].iloc[0]
    assert age.pairs == 1
    assert age.mean_absolute_change == pytest.approx(0.1)
