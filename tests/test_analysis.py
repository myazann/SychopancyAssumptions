"""Tests for `syco.analysis` -- the statistics behind `python -m syco analyze`.

The point of most of these is calibration rather than correctness of a formula:
a permutation test that ignores the design will happily report p < 0.001 on
data with no effect in it, and the only way to catch that is to feed it noise
and check that the p-values come back uniform.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from syco import analysis as an
from syco.data import NO_PERSONA

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import analyze_openended as driver  # noqa: E402


# ---------------------------------------------------------------------------
# small numeric helpers
# ---------------------------------------------------------------------------
def test_normalized_entropy_is_one_for_uniform_and_zero_for_a_point_mass():
    assert an.normalized_entropy([0.25] * 4) == pytest.approx(1.0)
    assert an.normalized_entropy([1.0, 0.0, 0.0]) == pytest.approx(0.0)
    assert an.normalized_entropy([0.5, 0.5]) > an.normalized_entropy([0.9, 0.1])


def test_jensen_shannon_spans_zero_to_one_bit():
    assert an.jensen_shannon([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.0)
    assert an.jensen_shannon([1, 0], [0, 1]) == pytest.approx(1.0)
    assert 0 < an.jensen_shannon([0.7, 0.3], [0.3, 0.7]) < 1


def test_log_odds_ratio_stays_finite_on_an_empty_cell():
    estimate, se = an.log_odds_ratio(0, 10, 5, 5)
    assert np.isfinite(estimate) and np.isfinite(se)
    assert estimate < 0                       # the target under-uses it


def test_mcnemar_is_exact_and_symmetric():
    assert an.mcnemar(0, 0) == 1.0
    assert an.mcnemar(10, 2) == pytest.approx(an.mcnemar(2, 10))
    assert an.mcnemar(20, 0) < 0.001


def test_cramers_v_is_zero_for_independence_and_one_for_a_perfect_split():
    assert an.cramers_v([[50, 50], [50, 50]]) == pytest.approx(0.0)
    assert an.cramers_v([[50, 0], [0, 50]]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# blocked permutation: the facet and framing contrasts
# ---------------------------------------------------------------------------
def _balanced_design(n_blocks=200, n_levels=4, n_features=3, seed=0):
    rng = np.random.default_rng(seed)
    block = np.repeat(np.arange(n_blocks), n_levels)
    level = np.tile(np.arange(n_levels), n_blocks)
    # A strong per-block effect and no level effect: exactly the shape the
    # design has, where the dilemma dominates and the facet is the question.
    block_mean = rng.random((n_blocks, n_features)) * 0.8 + 0.1
    values = (rng.random((n_blocks * n_levels, n_features))
              < block_mean[block]).astype(float)
    return values, block, level


def test_blocked_permutation_is_calibrated_when_only_the_block_matters():
    """Noise in, uniform p out -- even with a huge between-block effect.

    This is the whole reason the test is blocked. A test that pooled across
    blocks would read the block effect as a level effect and reject constantly.
    """
    p_values = []
    for seed in range(20):
        values, block, level = _balanced_design(seed=seed)
        observed, null, _ = an.blocked_permutation(
            values, block, level, n_perm=200, seed=seed)
        p_values.append(
            an.omnibus_from_permutation(observed[0], null[:, 0])["p_permutation"])
    # Under the null these are uniform; allow generous slack for 20 draws.
    assert np.mean(np.array(p_values) < 0.05) <= 0.2
    assert 0.25 < np.mean(p_values) < 0.75


def test_blocked_permutation_finds_a_real_level_effect():
    values, block, level = _balanced_design(seed=7)
    values[level == 0, 0] = 1.0                      # level 0 always states it
    observed, null, counts = an.blocked_permutation(
        values, block, level, n_perm=200, seed=7)
    report = an.omnibus_from_permutation(observed[0], null[:, 0])
    assert report["p_permutation"] < 0.01

    table = an.contrast_table(observed[0], null[:, 0], counts[0],
                              [f"L{i}" for i in range(4)],
                              [f"F{j}" for j in range(3)])
    hit = table[(table["level"] == "L0") & (table["feature"] == "F0")].iloc[0]
    assert hit["rate"] == pytest.approx(1.0)
    assert hit["log_or"] > 0
    assert hit["q_normal"] < 0.01


def test_blocked_permutation_drops_incomplete_blocks():
    values, block, level = _balanced_design(n_blocks=50, seed=1)
    keep = ~((block == 0) & (level == 0))            # block 0 loses a level
    observed, _, counts = an.blocked_permutation(
        values[keep], block[keep], level[keep], n_perm=20, seed=1)
    assert counts[0][0] == 49
    assert observed.shape == (1, 4, 3)


def test_blocked_permutation_groups_share_one_set_of_shuffles():
    values, block, level = _balanced_design(n_blocks=100, seed=3)
    group = np.where(block % 2 == 0, 0, 1)
    observed, null, counts = an.blocked_permutation(
        values, block, level, group=group, n_perm=30, seed=3)
    assert observed.shape == (2, 4, 3)
    assert null.shape == (30, 2, 4, 3)
    assert counts.tolist() == [[50] * 4, [50] * 4]
    # Every block is in exactly one group, so the groups sum to the pooled table.
    pooled, _pooled_null, _ = an.blocked_permutation(
        values, block, level, n_perm=30, seed=3)
    assert np.allclose(observed.sum(axis=0), pooled[0])


# ---------------------------------------------------------------------------
# cluster permutation: the demographic contrasts
# ---------------------------------------------------------------------------
def test_cluster_permutation_counts_people_not_responses():
    """A per-person attribute over noise must not become significant just
    because each person carries a thousand rows.

    The naive response-level test is the failure mode this guards: give 25
    people a random label and 400 rows each, and shuffling rows finds an effect
    every time while shuffling people does not.
    """
    rng = np.random.default_rng(11)
    n_clusters, per_cluster, n_features = 25, 400, 4
    cluster = np.repeat(np.arange(n_clusters), per_cluster)
    # Each person has their own base rate -- real between-person variation,
    # unrelated to the attribute.
    base = rng.random((n_clusters, n_features)) * 0.5 + 0.2
    values = (rng.random((n_clusters * per_cluster, n_features))
              < base[cluster]).astype(float)

    p_values = []
    for seed in range(15):
        levels = np.random.default_rng(seed).integers(0, 3, n_clusters)
        observed, null, _ = an.cluster_permutation(
            values, cluster, levels, n_perm=200, seed=seed)
        p_values.append(
            an.omnibus_from_permutation(observed, null)["p_permutation"])
    assert np.mean(np.array(p_values) < 0.05) <= 0.2

    # The asymptotic test on the same table is the one that over-rejects.
    levels = np.random.default_rng(0).integers(0, 3, n_clusters)
    observed, null, _ = an.cluster_permutation(values, cluster, levels,
                                               n_perm=200, seed=0)
    report = an.omnibus_from_permutation(observed, null)
    assert report["p_permutation"] > report["p_asymptotic"]


def test_cluster_permutation_finds_an_attribute_that_really_splits_people():
    rng = np.random.default_rng(5)
    n_clusters, per_cluster = 30, 200
    cluster = np.repeat(np.arange(n_clusters), per_cluster)
    levels = np.arange(n_clusters) % 2
    rate = np.where(levels == 0, 0.2, 0.8)
    values = (rng.random((n_clusters * per_cluster, 1))
              < rate[cluster][:, None]).astype(float)
    values = np.column_stack([values, 1 - values])
    observed, null, _ = an.cluster_permutation(values, cluster, levels,
                                               n_perm=400, seed=5)
    assert an.omnibus_from_permutation(observed, null)["p_permutation"] < 0.01


# ---------------------------------------------------------------------------
# stratified permutation: the sycophancy terciles
# ---------------------------------------------------------------------------
def test_stratified_permutation_holds_group_sizes_and_is_calibrated():
    rng = np.random.default_rng(2)
    n = 3000
    stratum = rng.integers(0, 15, n)
    level = rng.integers(0, 3, n)
    # A strong stratum effect, no level effect.
    stratum_rate = rng.random(15) * 0.8 + 0.1
    values = (rng.random((n, 4)) < stratum_rate[stratum][:, None]).astype(float)
    observed, null, counts = an.stratified_permutation(values, stratum, level,
                                                       n_perm=300, seed=2)
    assert counts.sum() == n
    assert np.allclose(observed.sum(axis=0), values.sum(axis=0))
    # Every shuffle keeps the level totals, so the grand total never moves.
    assert np.allclose(null.sum(axis=1), values.sum(axis=0), atol=1e-8)
    assert an.omnibus_from_permutation(observed, null)["p_permutation"] > 0.05


def test_stratified_permutation_finds_a_level_effect_inside_strata():
    rng = np.random.default_rng(4)
    n = 3000
    stratum = rng.integers(0, 15, n)
    level = rng.integers(0, 3, n)
    values = (rng.random((n, 3)) < 0.3).astype(float)
    values[level == 2, 0] = 1.0
    observed, null, _ = an.stratified_permutation(values, stratum, level,
                                                  n_perm=300, seed=4)
    assert an.omnibus_from_permutation(observed, null)["p_permutation"] < 0.01


# ---------------------------------------------------------------------------
# p-values
# ---------------------------------------------------------------------------
def test_empirical_p_never_returns_zero():
    null = np.zeros((100, 2))
    assert an.empirical_p(np.array([99.0, 99.0]), null).min() == pytest.approx(1 / 101)


def test_permutation_pvalues_escapes_the_monte_carlo_floor():
    """The reason the corrected tables use `p_normal`.

    With a few hundred draws the Monte-Carlo p-value bottoms out, and a family
    of thousands of cells then has no attainable significant q-value at all.
    The z against the same draws has no floor.
    """
    rng = np.random.default_rng(0)
    null = rng.normal(100.0, 5.0, (200, 1))
    result = an.permutation_pvalues(np.array([140.0]), null)
    assert result["p_mc"][0] == pytest.approx(1 / 201)
    assert result["p_normal"][0] < 1e-10
    assert result["usable"][0]


def test_permutation_pvalues_refuses_the_approximation_on_rare_cells():
    null = np.zeros((100, 2))
    null[:, 0] = np.random.default_rng(1).integers(0, 2, 100)   # mean ~0.5
    result = an.permutation_pvalues(np.array([3.0, 0.0]), null)
    assert not result["usable"].any()          # too rare, and a constant null
    assert np.isnan(result["p_normal"]).all()
    assert np.isfinite(result["p_mc"]).all()


def test_contrast_table_leaves_untestable_cells_without_a_q_value():
    rng = np.random.default_rng(3)
    observed = np.array([[200.0, 3.0], [180.0, 2.0]])
    null = np.stack([observed + rng.normal(0, 6, observed.shape)
                     for _ in range(300)])
    denominators = np.array([[203.0, 203.0], [182.0, 182.0]])
    table = an.contrast_table(observed, null, denominators,
                              ["a", "b"], ["common", "rare"])
    rare = table[table["feature"] == "rare"]
    assert not rare["testable"].any()
    assert rare["q_normal"].isna().all()
    assert rare["p_monte_carlo"].notna().all()


def test_benjamini_hochberg_is_monotone_and_preserves_nan():
    adjusted = an.benjamini_hochberg([0.001, 0.01, 0.2, 0.9])
    assert (np.diff(adjusted) >= -1e-12).all()
    assert adjusted.max() <= 1.0
    assert np.isnan(an.benjamini_hochberg([np.nan, 0.01])[0])


# ---------------------------------------------------------------------------
# shaping the corpus
# ---------------------------------------------------------------------------
def _tiny_assumptions() -> pd.DataFrame:
    rows = []
    for model in ("A", "B"):
        for persona in (NO_PERSONA, "p1", "p2"):
            ptype = NO_PERSONA if persona == NO_PERSONA else "hobbies"
            for framing in ("original_post", "flipped_story"):
                for rank in range(3):
                    rows.append({
                        "model": model, "persona_type": ptype,
                        "persona_id": persona, "prompt_type": framing,
                        "prompt_id": "q1", "rep": 0, "rank": rank,
                        "assumption": f"label {rank}",
                        "description": "text",
                        "probability_norm": [0.5, 0.3, 0.2][rank],
                        "parse_status": "clean",
                        "topic": rank % 2,
                        "_text": f"label {rank}",
                    })
    return pd.DataFrame(rows)


def test_response_table_summarizes_each_completion_once():
    frame = _tiny_assumptions()
    responses = an.response_table(frame)
    assert len(responses) == 2 * 3 * 2
    assert (responses["n_assumptions"] == 3).all()
    assert (responses["top1_prob"] == 0.5).all()
    assert (responses["n_topics"] == 2).all()
    assert 0 < responses["prob_entropy"].iloc[0] < 1


def test_response_table_survives_a_missing_probability():
    """One unparsed probability must not void a whole response vector.

    It happened: two assumptions in 60,000 parsed without one, and the NaN
    propagated through the weighted mean into the control response for a whole
    dilemma, which then voided the 250 distances measured against it.
    """
    frame = _tiny_assumptions()
    frame.loc[frame.index[:3], "probability_norm"] = np.nan
    embeddings = np.tile(np.arange(4.0), (len(frame), 1)).astype("float32")
    embeddings += np.arange(len(frame))[:, None]
    responses = an.response_table(frame, embeddings)
    assert np.isfinite(responses.attrs["vectors"]).all()


def test_topic_counts_rows_sum_to_the_assumptions_of_that_response():
    frame = _tiny_assumptions()
    responses = an.response_table(frame)
    corpus = an.Corpus(assumptions=frame, responses=responses,
                       topics=pd.DataFrame({"topic": [0, 1], "name": ["x", "y"]}),
                       indicators=np.zeros((len(responses), 2)),
                       topic_ids=[0, 1])
    counts = an.topic_counts(corpus)
    assert counts.shape == (len(responses), 2)
    assert (counts.sum(axis=1) == 3).all()


def test_distance_to_control_pairs_on_dilemma_and_framing():
    frame = _tiny_assumptions()
    responses = an.response_table(frame)
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(len(responses), 8)).astype("float32")
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    corpus = an.Corpus(assumptions=frame, responses=responses,
                       topics=pd.DataFrame({"topic": [0, 1], "name": ["x", "y"]}),
                       indicators=np.zeros((len(responses), 2)),
                       topic_ids=[0, 1], vectors=vectors)
    distance = an.distance_to_control(corpus)
    control = responses["persona_type"] == NO_PERSONA
    assert distance[control.to_numpy()].isna().all()
    assert distance[~control.to_numpy()].notna().all()
    assert (distance.dropna() >= 0).all()


def test_pairwise_reference_distances_matches_the_explicit_pairwise_mean():
    """The closed form has to equal the matrix it replaces.

    For unit vectors the off-diagonal similarities sum to ||sum(v)||^2 - n, so
    a group's mean pairwise distance comes from one grouped add. Building the
    matrices instead cost four minutes on the real corpus; this pins the
    shortcut to the thing it stands in for.
    """
    rng = np.random.default_rng(9)
    rows, vectors = [], []
    for model in ("A", "B"):
        for person in ("p1", "p2", "p3"):
            for facet in ("hobbies", "politics", "family"):
                for framing in ("original_post", "flipped_story"):
                    rows.append({"model": model, "persona_type": facet,
                                 "persona_id": person, "prompt_type": framing,
                                 "prompt_id": "q1", "rep": 0,
                                 "topics": (0,)})
                    vectors.append(rng.normal(size=6))
    responses = pd.DataFrame(rows)
    matrix = np.array(vectors)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    # The corpus stores float32, so compare against the same precision the
    # function will see rather than against the float64 originals.
    matrix = matrix.astype("float32").astype("float64")
    corpus = an.Corpus(assumptions=pd.DataFrame(), responses=responses,
                       topics=pd.DataFrame({"topic": [0], "name": ["x"]}),
                       indicators=np.zeros((len(responses), 1)),
                       topic_ids=[0], vectors=matrix.astype("float32"))
    table = an.pairwise_reference_distances(corpus).set_index(
        ["model", "comparison"])["mean_distance"]

    # the same quantity, computed the slow explicit way
    def explicit(group_keys):
        totals, weights = {}, {}
        for key, block in responses.groupby(group_keys, sort=False):
            index = block.index.to_numpy()
            if len(index) < 2:
                continue
            similarity = matrix[index] @ matrix[index].T
            upper = np.triu_indices(len(index), k=1)
            model = block["model"].iat[0]
            totals[model] = totals.get(model, 0.0) + (1 - similarity[upper]).sum()
            weights[model] = weights.get(model, 0) + len(upper[0])
        return {m: totals[m] / weights[m] for m in totals}

    for comparison, keys in (
        ("same_person_other_facet",
         ["model", "persona_id", "prompt_id", "prompt_type"]),
        ("other_person_same_facet",
         ["model", "persona_type", "prompt_id", "prompt_type"]),
        ("cross_framing",
         ["model", "persona_type", "persona_id", "prompt_id"]),
    ):
        for model, value in explicit(keys).items():
            assert table[(model, comparison)] == pytest.approx(value, abs=1e-7)


def test_pairwise_reference_distances_ignores_a_broken_vector():
    responses = pd.DataFrame([
        {"model": "A", "persona_type": f, "persona_id": "p1",
         "prompt_type": "original_post", "prompt_id": "q1", "rep": 0}
        for f in ("hobbies", "politics", "family")])
    matrix = np.eye(3, 4, dtype="float32")
    matrix[2] = np.nan
    corpus = an.Corpus(assumptions=pd.DataFrame(), responses=responses,
                       topics=pd.DataFrame({"topic": [0], "name": ["x"]}),
                       indicators=np.zeros((3, 1)), topic_ids=[0],
                       vectors=matrix)
    table = an.pairwise_reference_distances(corpus)
    assert np.isfinite(table["mean_distance"]).all()


def test_detectable_difference_scales_with_the_null_spread():
    frame = pd.DataFrame({
        "level": ["a", "a", "b", "b"],
        "n": [1000.0] * 4,
        "rate": [0.1, 0.2, 0.1, 0.2],
        "delta": [0.01, -0.01, -0.01, 0.01],
        "null_sd": [5.0, 5.0, 20.0, 20.0],
        "testable": [True, True, True, True],
    })
    out = an.detectable_difference(frame, group_fields=["level"]).set_index("level")
    assert out.loc["b", "min_detectable_delta_median_topic"] == pytest.approx(
        4 * out.loc["a", "min_detectable_delta_median_topic"])
    # (1.96 + 0.8416) * 5 / 1000
    assert out.loc["a", "min_detectable_delta_median_topic"] == pytest.approx(
        0.01401, abs=1e-4)


def test_detectable_difference_skips_untestable_rows():
    frame = pd.DataFrame({
        "level": ["a", "a"], "n": [1000.0, 1000.0], "rate": [0.1, 0.2],
        "delta": [0.01, -0.01], "null_sd": [5.0, 500.0],
        "testable": [True, False]})
    out = an.detectable_difference(frame, group_fields=["level"])
    assert out["topics_testable"].iloc[0] == 1


def test_topic_indicators_count_a_repeated_topic_once():
    responses = pd.DataFrame({"topics": [(0, 1), (1,), ()]})
    matrix, topics = an.topic_indicators(responses)
    assert topics == [0, 1]
    assert matrix.tolist() == [[1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]


# ---------------------------------------------------------------------------
# demographics
# ---------------------------------------------------------------------------
def _demographics() -> pd.DataFrame:
    return pd.DataFrame({
        "persona_id": [f"p{i}" for i in range(10)],
        "country": ["USA"] * 10,                       # one level: unusable
        "gender": ["M"] * 6 + ["F"] * 4,               # usable
        "rare": ["a"] * 9 + ["b"],                     # one tiny level
        "vulnerability_score": [0, 1, 1, 2, 2, 3, 3, 4, 5, 6],
    })


def test_demographic_columns_records_why_each_was_dropped():
    coverage = an.demographic_columns(_demographics(),
                                      [f"p{i}" for i in range(10)],
                                      min_per_level=3)
    by_column = coverage.set_index("column")
    assert by_column.loc["gender", "usable"]
    assert not by_column.loc["country", "usable"]
    assert "1 distinct value" in by_column.loc["country", "reason_dropped"]
    assert not by_column.loc["rare", "usable"]
    assert by_column.loc["rare", "reason_dropped"]


def test_demographic_columns_screens_only_the_sampled_personas():
    """A column can be fine over 200 people and degenerate over the 25 drawn."""
    demo = _demographics()
    coverage = an.demographic_columns(demo, ["p0", "p1", "p2", "p3"],
                                      min_per_level=3)
    assert not coverage.set_index("column").loc["gender", "usable"]


def test_collapse_levels_drops_the_levels_it_cannot_compare():
    values = pd.Series(["a"] * 5 + ["b"] * 4 + ["c"])
    collapsed = an.collapse_levels(values, min_per_level=3)
    assert set(collapsed.dropna()) == {"a", "b"}
    assert collapsed.isna().sum() == 1


def test_demographic_columns_never_offers_the_vulnerability_indices():
    """They are integer counts over 25 people; binning one is not an analysis."""
    demo = _demographics()
    coverage = an.demographic_columns(demo, [f"p{i}" for i in range(10)],
                                      min_per_level=3)
    assert "vulnerability_score" not in set(coverage["column"])
    assert set(an.VULNERABILITY_COLUMNS) <= an.DEMOGRAPHIC_SKIP


def _persona_corpus(seed=0):
    """A corpus where one word in the transcript really does track one topic."""
    rng = np.random.default_rng(seed)
    rows, texts = [], []
    for person in range(30):
        loud = person < 15
        texts.append({
            "persona_type": "hobbies", "persona_id": f"p{person}",
            "persona_text": ("i climb mountains every weekend " * 6
                             if loud else "i read quietly at home " * 6),
        })
        for prompt in range(20):
            # the loud half gets topic 0 far more often
            topic = 0 if (loud and rng.random() < 0.8) else 1
            for rank in range(3):
                rows.append({
                    "model": "A", "persona_type": "hobbies",
                    "persona_id": f"p{person}", "prompt_type": "original_post",
                    "prompt_id": f"q{prompt}", "rep": 0, "rank": rank,
                    "assumption": f"label {topic}", "description": "text",
                    "probability_norm": [0.5, 0.3, 0.2][rank],
                    "parse_status": "clean", "topic": topic if rank == 0 else 1,
                    "_text": f"label {topic}",
                })
    assumptions = pd.DataFrame(rows)
    responses = an.response_table(assumptions)
    corpus = an.Corpus(
        assumptions=assumptions, responses=responses,
        topics=pd.DataFrame({"topic": [0, 1], "name": ["climbing", "reading"]}),
        indicators=np.zeros((len(responses), 2)), topic_ids=[0, 1])
    return corpus, pd.DataFrame(texts)


def test_persona_words_by_topic_finds_a_word_that_tracks_a_topic():
    corpus, personas = _persona_corpus()
    words = an.persona_words_by_topic(corpus, personas, min_units=5)
    assert not words.empty
    climbing = words[words["topic"].str.contains("climbing")]
    top = climbing.reindex(
        climbing["z"].abs().sort_values(ascending=False).index).iloc[0]
    assert top["word"] in {"climb", "mountains", "weekend", "every"}
    assert top["z"] > 0                     # over-used where the topic is common
    assert top["above_threshold"]
    assert top["topic_rate_often"] > top["topic_rate_rarely"]


def test_persona_words_by_topic_returns_nothing_without_enough_units():
    corpus, personas = _persona_corpus()
    assert an.persona_words_by_topic(corpus, personas, min_units=500).empty


def test_fightin_words_is_signed_toward_the_target():
    target = np.array([50.0, 1.0])
    reference = np.array([1.0, 50.0])
    prior = target + reference
    z = an._fightin_words(target, reference, prior)
    assert z[0] > 0 and z[1] < 0


def test_load_persona_texts_keeps_only_the_persons_own_turns(tmp_path):
    path = tmp_path / "personas.gz"
    transcript = ('[{"role": "user", "content": "i love birdwatching"}, '
                  '{"role": "assistant", "content": "how wonderful for you"}]')
    pd.DataFrame({"persona_type": ["hobbies"], "persona_id": ["p1"],
                  "persona_text": [transcript]}).to_pickle(path)
    out = an.load_persona_texts(path)
    assert out.loc[0, "persona_text"] == "i love birdwatching"
    assert "wonderful" not in out.loc[0, "persona_text"]


def test_order_facets_keeps_the_source_order_not_the_alphabet():
    ordered = an.order_facets(["politics", "assumptions", "hobbies"])
    assert ordered.index("hobbies") < ordered.index("politics")
    assert ordered[-1] == "assumptions"
    assert an.order_facets(["zzz", "hobbies"]) == ["hobbies", "zzz"]


def test_load_demographics_keeps_an_occupation_code_categorical(tmp_path):
    path = tmp_path / "demo.csv"
    pd.DataFrame({"uuid": ["a", "b"], "major_group_code": [51, 53]}).to_csv(
        path, index=False)
    demo = an.load_demographics(path)
    assert "persona_id" in demo.columns
    assert not pd.api.types.is_numeric_dtype(demo["major_group_code"])


# ---------------------------------------------------------------------------
# the driver's own small helpers
# ---------------------------------------------------------------------------
def test_flags_survives_a_csv_round_trip(tmp_path):
    """A boolean column read back as text must not select every row.

    `read_csv` hands these back as real booleans most of the time and as the
    strings "True"/"False" under the Arrow string backend; indexing a frame by
    the string version silently keeps everything, which reads as "all of them
    were significant".
    """
    frame = pd.DataFrame({"above": [True, False, True], "z": [3.0, 0.1, 2.5]})
    path = tmp_path / "flags.csv"
    frame.to_csv(path, index=False)
    for table in (frame, pd.read_csv(path),
                  pd.DataFrame({"above": ["True", "False", "True"]})):
        assert driver._flags(table, "above").tolist() == [True, False, True]


def test_short_breaks_a_topic_name_on_a_separator():
    assert driver._short("T3 trauma / trauma informed / past") == \
        "trauma / trauma informed"
    assert "/" not in driver._short("T1 perfectionist", chars=40)


def test_markdown_table_renders_an_underflowed_p_honestly():
    frame = pd.DataFrame({"p_asymptotic": [0.0], "chi2": [6912.0]})
    rendered = driver._markdown_table(frame)
    assert "<1e-308" in rendered
    assert "| 0 |" not in rendered


def test_discover_scores_pairs_a_model_with_its_own_collection(
        tmp_path, monkeypatch):
    """And keeps it away from a collection belonging to another model."""
    # Keep this discovery test independent of real result files in a populated
    # checkout; `extra_dirs` augments the production search path by design.
    monkeypatch.setattr(driver, "SCORE_DIRECTORIES", ())
    (tmp_path / "gemma-3-12b-it_binary_sycophancy.parquet").touch()
    (tmp_path / "gemma-3-12b-it_long_results.pkl").touch()
    models = ["Gemma3-12B", "Llama-3.1-8B"]
    found = driver.discover_scores(models, extra_dirs=[str(tmp_path)])
    assert "Llama-3.1-8B" not in found          # skipped, not borrowed
    name, path = found["Gemma3-12B"]
    assert name == "gemma-3-12b-it"
    # the long-form table holds replies, not the constrained log-probabilities
    assert "long" not in path.name


def test_discover_scores_rejects_an_alias_that_is_not_analyzed(tmp_path):
    target = tmp_path / "x_results.pkl"
    target.touch()
    with pytest.raises(SystemExit):
        driver.discover_scores(["Gemma3-12B"], [f"NotAModel={target}"])
