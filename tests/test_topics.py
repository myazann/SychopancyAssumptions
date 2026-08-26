import math

import pandas as pd

from scripts.topic_assumptions import (
    _facet_terms,
    default_prefix,
    degenerate,
    trim,
)
from syco.data import NO_PERSONA
from syco.topics import (
    _clean_label,
    corpus_profile,
    ngram_frequencies,
    ngrams,
    segments,
    topic_entropy,
    topic_shares,
    topics_available,
)


def test_bigrams_do_not_span_punctuation_but_do_span_hyphens():
    assert segments("Seeking validation. They're people-pleasing") == [
        ["seeking", "validation"], ["they're", "people", "pleasing"]
    ]
    grams = ngrams("Seeking validation. They're people-pleasing", n=2)
    assert "seeking validation" in grams
    assert "people pleasing" in grams
    assert "validation they're" not in grams


def test_a_term_repeated_in_one_assumption_is_counted_once():
    assert ngrams("validation and more validation") == [
        "validation", "and", "more"
    ]


def test_unigrams_drop_function_words_and_bigrams_keep_them():
    assert ngrams("the user is seeking validation", n=1, drop_stopwords=True) == [
        "user", "seeking", "validation"
    ]
    assert "rather than" in ngrams("helpful rather than honest", n=2)


def _frame(rows):
    """One tidy assumptions table; each row is (facet, person, k labels)."""
    out = []
    for facet, person, labels in rows:
        for rank, label in enumerate(labels):
            out.append({
                "run_id": "run-1", "probe": "openended3v2/native",
                "history_mode": "native", "persona_type": facet,
                "persona_id": person, "prompt_type": "original_post",
                "prompt_id": "q1", "rep": 0, "rank": rank,
                "assumption": label, "description": "",
            })
    return pd.DataFrame(out)


def test_word_and_bigram_shares_use_different_denominators():
    # One response, three assumptions, two of which say "validation".
    df = _frame([("politics", "p1", ["seeking validation", "wants validation",
                                     "curious reader"])])
    table = ngram_frequencies(df, by=(), min_count=1, top=None)
    overall = table[table.scope == "overall"]
    word = overall[(overall.level == "unigram") & (overall.term == "validation")].iloc[0]

    assert word.n_assumptions == 2 and word.assumptions == 3
    assert word.share_assumptions == 2 / 3
    # The same word covers the single response completely: the paper quotes
    # words per assumption and bigrams per response, and they differ by k.
    assert word.n_responses == 1 and word.responses == 1
    assert word.share_responses == 1.0


def test_facet_term_lift_is_measured_against_the_persona_free_control():
    df = _frame([
        (NO_PERSONA, NO_PERSONA, ["seeking validation", "curious reader"]),
        ("politics", "p1", ["seeking validation", "seeking validation"]),
    ])
    table = ngram_frequencies(df, by=("persona_type",), min_count=1, top=None)
    facets = table[(table.scope == "persona_type") & (table.level == "unigram")]
    politics = facets[(facets.group == "politics") & (facets.term == "validation")].iloc[0]

    assert politics.share_assumptions == 1.0
    assert politics.control_share == 0.5
    assert politics.lift == 2.0


def test_terms_absent_from_the_control_have_no_lift_rather_than_infinity():
    df = _frame([
        (NO_PERSONA, NO_PERSONA, ["curious reader"]),
        ("politics", "p1", ["seeking validation"]),
    ])
    table = ngram_frequencies(df, by=("persona_type",), min_count=1, top=None)
    row = table[(table.scope == "persona_type") & (table.group == "politics") &
                (table.term == "validation")].iloc[0]
    assert math.isnan(row.lift)


def test_min_count_does_not_erase_the_control_denominator():
    # "validation" appears once in the control and three times under politics.
    # Filtering before dividing would drop the control row and report no lift
    # on the term the facet raised most.
    df = _frame([
        (NO_PERSONA, NO_PERSONA, ["seeking validation", "curious reader",
                                  "curious reader", "curious reader"]),
        ("politics", "p1", ["seeking validation", "seeking validation",
                            "seeking validation", "curious reader"]),
    ])
    table = ngram_frequencies(df, by=("persona_type",), min_count=2, top=None)
    row = table[(table.scope == "persona_type") & (table.group == "politics") &
                (table.term == "validation")].iloc[0]
    assert row.control_share == 0.25
    assert row.lift == 3.0


def test_facet_terms_fall_back_to_share_when_a_run_has_no_control():
    df = _frame([("politics", "p1", ["seeking validation", "curious reader"])])
    table = ngram_frequencies(df, by=("persona_type",), min_count=1, top=None)
    ranked, note = _facet_terms(table, top=5)
    assert not ranked.empty
    assert "no 'none' control cells" in note


def test_corpus_profile_separates_responses_from_assumptions():
    df = _frame([
        ("politics", "p1", ["a reader", "a writer"]),
        ("politics", "p2", ["a reader", "a doubter"]),
    ])
    df.loc[df.persona_id == "p2", "prompt_id"] = "q2"
    profile = corpus_profile(df).iloc[0]
    assert profile.assumptions == 4
    assert profile.responses == 2
    assert profile.distinct_labels == 3


def test_topic_shares_sum_to_one_within_a_facet_and_lift_uses_the_control():
    df = _frame([
        (NO_PERSONA, NO_PERSONA, ["a", "b", "c", "d"]),
        ("politics", "p1", ["a", "b", "c", "d"]),
    ])
    # Control splits 50/50 across two topics; politics is entirely topic 0.
    assignments = pd.Series([0, 0, 1, 1, 0, 0, 0, 0], index=df.index)
    shares = topic_shares(df, assignments)

    for facet, group in shares.groupby("persona_type"):
        assert abs(group.share_assumptions.sum() - 1.0) < 1e-9, facet
    politics = shares[(shares.persona_type == "politics") & (shares.topic == 0)].iloc[0]
    assert politics.share_assumptions == 1.0
    assert politics.lift == 2.0

    entropy = topic_entropy(shares).set_index("persona_type")
    assert entropy.loc[NO_PERSONA, "entropy_bits"] == 1.0
    assert entropy.loc["politics", "entropy_bits"] == 0.0


def test_topics_available_reports_a_reason_when_it_is_not():
    available, why = topics_available()
    assert available or "pip install" in why


def test_llm_topic_labels_are_stripped_of_the_decoration_models_add():
    assert _clean_label('**Label:** "Seeking validation"\n', 5) == "Seeking validation"
    assert _clean_label("<think>hmm</think>\nEmotional support", 5) == "Emotional support"
    assert _clean_label("", 5) == "(no label)"
    assert _clean_label("one two three four five six", 5) == "one two three four five"


def test_printing_trims_joined_columns_without_touching_the_written_table():
    info = pd.DataFrame([{"topic": 0, "top_words": "a, b, c, d",
                          "examples": "one | two | three"}])
    trimmed = trim(info, top_words=2, examples=1)
    assert trimmed.top_words.iloc[0] == "a, b"
    assert trimmed.examples.iloc[0] == "one"
    assert info.top_words.iloc[0] == "a, b, c, d"


def _info(shares):
    return pd.DataFrame([{"topic": t, "share_assumptions": s}
                         for t, s in shares.items()])


def test_a_topic_model_that_found_no_structure_says_so():
    assert "fewer than two topics" in degenerate(_info({0: 1.0}), 0.0)[0]
    assert "holds 90%" in degenerate(_info({0: 0.9, 1: 0.1}), 0.0)[0]
    assert "outliers" in degenerate(_info({-1: 0.6, 0: 0.2, 1: 0.2}), 0.6)[-1]
    assert degenerate(_info({0: 0.4, 1: 0.35, 2: 0.25}), 0.1) == []


def test_output_prefix_drops_the_parser_suffix():
    assert default_prefix("results/run_assumptions.parquet").name == "run"
