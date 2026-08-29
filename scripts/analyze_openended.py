#!/usr/bin/env python3
"""The three analyses of the open-ended assumptions grid, in one pass.

    python -m syco analyze --model Llama-3.1-8B --model Qwen3.6-35B-A3B

Writes a directory of results rather than printing them, because all three
questions produce tables that are read next to each other:

    01_persona_framing/   what the disclosed facet and the story's telling do
                          to the assumptions
    02_persona_text/      which words in a person's own transcript go with
                          what the model assumed about them
    03_sycophancy/        which assumptions travel with each model's own
                          forced-choice sycophancy score

    KEY_FINDINGS.md       the numbers that survived their own tests
    README.md             what every file is and how to read it

Each section writes its statistical tables as CSV, its figures as PNG, and a
short markdown that says what the tables show. The CSV is the figure's table
view: nothing is only in a picture.

One topic space is fitted over every model's assumptions pooled, so a topic
means the same thing in all three sections and across models. Fitting per
section would leave the sections uncomparable.

The three sections use three different null hypotheses, because the design
gives each a different unit of independence -- see `syco.analysis`. The short
version: facet and framing are tested by shuffling labels inside each
person-and-dilemma block, and sycophancy by shuffling inside each dilemma.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from syco import analysis as an
from syco import figures as viz
from syco import report as rp
from syco import sycophancy as syc
from syco.data import FLIPPED, NO_PERSONA, ORIGINAL
from syco.topics import PROMPT_ECHO, ngram_frequencies

#: Joins the two factors into one grouping key. A pipe rather than a space:
#: persona type names contain underscores, prompt types contain underscores,
#: and the two have to come apart again on the page.
SLICE_SEP = " | "

RESULTS = Path("results")
DEFAULT_OUT = RESULTS / "analysis"
DEFAULT_PROBE = "openended3"

#: How many topics a heatmap or a top-list shows before it stops being
#: readable. The full table is always in the CSV beside it.
TOP_TOPICS = 18
TOP_N = 5


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _write(frame: pd.DataFrame, path: Path, index: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=index)
    return path


def _say(message: str) -> None:
    print(message, flush=True)


def _codes(values) -> tuple[np.ndarray, list]:
    """Integer codes and their ordered levels, for the permutation engine."""
    categories = pd.Categorical(values)
    return categories.codes.astype(int), list(categories.categories)


def _ordered_codes(values, order) -> tuple[np.ndarray, list]:
    categories = pd.Categorical(values, categories=list(order), ordered=True)
    return categories.codes.astype(int), list(order)


def _denominators(observed: np.ndarray) -> np.ndarray:
    """Row totals broadcast over the columns.

    Each assumption has exactly one topic, so a level's row total is the number
    of assumptions it verbalized -- the denominator every rate on the table is
    a share of.
    """
    return np.repeat(observed.sum(axis=-1, keepdims=True), observed.shape[-1],
                     axis=-1)


def _topic_labels(corpus: an.Corpus) -> dict:
    return {int(topic): f"T{int(topic)} {name}"
            for topic, name in zip(corpus.topics["topic"], corpus.topics["name"])}


def _rank_topics(corpus: an.Corpus, limit: int = TOP_TOPICS) -> list:
    """The largest topics, outliers excluded -- what a figure can show."""
    info = corpus.topics[corpus.topics["topic"] >= 0]
    return [int(t) for t in
            info.sort_values("n_assumptions", ascending=False)["topic"].head(limit)]


def _null_band(observed: np.ndarray, null: np.ndarray, low: float = 2.5,
               high: float = 97.5) -> tuple[np.ndarray, np.ndarray]:
    """The central 95% of what the shuffles produced, per cell.

    Drawn instead of a Wald interval around the estimate. A confidence interval
    for a share computed over k=3 correlated assumptions per response would be
    too narrow by an unknown factor; the permutation null needs no independence
    assumption, and "the dot is outside the band" is the same statement the
    p-value on the table makes.
    """
    return (np.percentile(null, low, axis=0), np.percentile(null, high, axis=0))


def _short(label: str, words: int = 3, chars: int = 30) -> str:
    """A topic's handle for an axis tick: its leading c-TF-IDF terms.

    Truncating the joined string cuts mid-word ("trauma informed / pas"); the
    terms are already ranked, so dropping whole ones from the end keeps the
    label a phrase.
    """
    body = label.split(" ", 1)[-1]
    terms, out = [t.strip() for t in body.split("/")], []
    for term in terms[:words]:
        candidate = " / ".join(out + [term])
        if out and len(candidate) > chars:
            break
        out.append(term)
    return " / ".join(out) or body[:chars]


def _flags(frame: pd.DataFrame, column: str) -> pd.Series:
    """A boolean column, however a CSV round-trip handed it back.

    `read_csv` returns these as real booleans most of the time and as the
    strings "True"/"False" under the Arrow string backend, and a frame indexed
    by the string version silently selects every row.
    """
    values = frame[column]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    return values.astype("string").str.lower().isin(["true", "1"])


def _cell(value, column: str = "") -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "-"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        value = float(value)
        # An asymptotic chi-square p can underflow to exactly 0 in float64.
        # Printing "0" claims certainty the arithmetic does not have.
        if value == 0.0 and _is_probability(column):
            return "<1e-308"
        magnitude = abs(value)
        if magnitude and (magnitude < 1e-3 or magnitude >= 1e6):
            return f"{value:.2e}"
        return f"{value:,.4g}"
    return str(value).replace("|", "\\|")


def _is_probability(column: str) -> bool:
    return (column.startswith(("p_", "q_"))
            or column.endswith(("_value", "_p", "_pvalue")))


def _markdown_table(frame: pd.DataFrame) -> str:
    """A GitHub table, without taking a dependency on `tabulate` for it."""
    if frame.empty:
        return "_(no rows)_"
    columns = list(frame.columns)
    lines = ["| " + " | ".join(str(c) for c in columns) + " |",
             "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(
            _cell(v, str(c)) for c, v in zip(columns, row)) + " |")
    return "\n".join(lines)


def _fmt_p(value) -> str:
    if not np.isfinite(value):
        return "n/a"
    return "<0.001" if value < 0.001 else f"{value:.3f}"


# ---------------------------------------------------------------------------
# 1. persona facet and story framing
# ---------------------------------------------------------------------------
def section_persona_framing(corpus: an.Corpus, out: Path, *, n_perm: int,
                            seed: int) -> dict:
    """What the disclosed facet and the telling do to the verbalized assumptions.

    Two contrasts, both within-subject and both tested by shuffling the label
    inside the block that holds everything else fixed:

    * facet -- block is (person, dilemma, framing), ten facets of one person;
    * framing -- block is (facet, person, dilemma), the two tellings.

    Splitting the facet shuffles by framing in one pass is what makes the
    interaction testable: the difference of two lifts needs both lifts computed
    on the same draw to have a null.
    """
    out.mkdir(parents=True, exist_ok=True)
    figures = out / "figures"
    figures.mkdir(exist_ok=True)

    responses = corpus.responses.reset_index(drop=True)
    counts = an.topic_counts(corpus)
    distance = an.distance_to_control(corpus).to_numpy()
    responses = responses.assign(dist_to_control=distance)
    labels = _topic_labels(corpus)
    topic_names = [labels[int(t)] for t in corpus.topic_ids]

    omnibus_rows, facet_rows, framing_rows, interaction_rows = [], [], [], []
    metric_rows, divergence_rows, mcnemar_rows = [], [], []
    written = []

    for model in corpus.models:
        model_mask = (responses["model"] == model).to_numpy()
        real = model_mask & (responses["persona_type"] != NO_PERSONA).to_numpy()

        # -- facet contrast, split by framing -------------------------------
        subset = responses[real]
        block, _ = _codes(list(zip(subset["persona_id"], subset["prompt_id"],
                                   subset["prompt_type"])))
        facets = an.order_facets(subset["persona_type"])
        level, facet_levels = _ordered_codes(subset["persona_type"], facets)
        framing_of_row, framing_levels = _ordered_codes(
            subset["prompt_type"], [ORIGINAL, FLIPPED])

        observed, null, block_counts = an.blocked_permutation(
            counts[real], block, level, group=framing_of_row,
            n_perm=n_perm, seed=seed)

        pooled_observed = observed.sum(axis=0)
        pooled_null = null.sum(axis=1)
        report = an.omnibus_from_permutation(pooled_observed, pooled_null)
        omnibus_rows.append({"model": model, "factor": "persona_type",
                             "framing": "both", "levels": len(facet_levels),
                             **report})
        for index, framing in enumerate(framing_levels):
            omnibus_rows.append({
                "model": model, "factor": "persona_type", "framing": framing,
                "levels": len(facet_levels),
                **an.omnibus_from_permutation(observed[index], null[:, index])})

        table = an.contrast_table(
            pooled_observed, pooled_null, _denominators(pooled_observed),
            facet_levels, topic_names, level_field="persona_type",
            feature_field="topic")
        table.insert(0, "model", model)
        facet_rows.append(table)

        # -- facet x framing interaction ------------------------------------
        rate_observed = observed / np.maximum(_denominators(observed), 1e-12)
        rate_null = null / np.maximum(_denominators(null), 1e-12)
        did_observed = rate_observed[0] - rate_observed[1]
        did_null = rate_null[:, 0] - rate_null[:, 1]
        did = an.permutation_pvalues(did_observed, did_null, min_expected=0.0)
        # The statistic here is a difference of two shares, not a count, so the
        # count-based usability rule does not apply; a cell is testable when
        # its null varied and both sides had assumptions to differ over.
        support = np.minimum(observed[0], observed[1])
        for i, facet in enumerate(facet_levels):
            for j, topic in enumerate(topic_names):
                interaction_rows.append({
                    "model": model, "persona_type": facet, "topic": topic,
                    "share_original": float(rate_observed[0, i, j]),
                    "share_flipped": float(rate_observed[1, i, j]),
                    "difference_original_minus_flipped": float(did_observed[i, j]),
                    "z": float(did["z"][i, j]),
                    "p_normal": float(did["p_normal"][i, j]),
                    "p_monte_carlo": float(did["p_mc"][i, j]),
                    "testable": bool(did["usable"][i, j] and support[i, j] >= 5),
                })

        # -- continuous response metrics, same blocks -----------------------
        # One permutation per metric, not one for all four. A block missing a
        # value for any metric is dropped from that permutation, and
        # `dist_to_control` is missing wherever a dilemma has no no-persona
        # response -- which on a partial run is several. Pooling the metrics
        # would let that hole silently shrink the sample behind the other
        # three, and their reported means would then be over a different set
        # of blocks than the row says.
        for metric in ("top1_prob", "prob_entropy", "n_topics",
                       "dist_to_control"):
            values = subset[[metric]].to_numpy("float64")
            try:
                observed_m, null_m, counts_m = an.blocked_permutation(
                    values, block, level, group=framing_of_row,
                    n_perm=n_perm, seed=seed)
            except ValueError:                  # no complete block for it
                continue
            for g, framing in enumerate(framing_levels):
                for i, facet in enumerate(facet_levels):
                    metric_rows.append({
                        "model": model, "prompt_type": framing,
                        "persona_type": facet, "metric": metric,
                        "n_responses": int(counts_m[g, i]),
                        "mean": float(observed_m[g, i, 0]
                                      / max(counts_m[g, i], 1)),
                        "p_permutation": float(an.empirical_p(
                            observed_m[g, i, 0], null_m[:, g, i, 0])),
                    })

        # -- framing contrast -----------------------------------------------
        model_rows = responses[model_mask]
        fblock, _ = _codes(list(zip(model_rows["persona_type"],
                                    model_rows["persona_id"],
                                    model_rows["prompt_id"])))
        flevel, framings = _ordered_codes(model_rows["prompt_type"],
                                          [ORIGINAL, FLIPPED])
        fobserved, fnull, fcounts = an.blocked_permutation(
            counts[model_mask], fblock, flevel, n_perm=n_perm, seed=seed)
        omnibus_rows.append({
            "model": model, "factor": "prompt_type", "framing": "both",
            "levels": 2,
            **an.omnibus_from_permutation(fobserved[0], fnull[:, 0])})
        ftable = an.contrast_table(
            fobserved[0], fnull[:, 0], _denominators(fobserved[0]),
            framings, topic_names, level_field="prompt_type",
            feature_field="topic")
        ftable.insert(0, "model", model)
        framing_rows.append(ftable)

        # Exact paired test on the same blocks, as the reading that does not
        # depend on how many shuffles were drawn.
        indicators, _ = an.topic_indicators(model_rows, corpus.topic_ids)
        paired = _paired_indicator_tensor(indicators, fblock, flevel)
        for j, topic in enumerate(topic_names):
            original_only = int(((paired[:, 0, j] > 0) & (paired[:, 1, j] == 0)).sum())
            flipped_only = int(((paired[:, 0, j] == 0) & (paired[:, 1, j] > 0)).sum())
            mcnemar_rows.append({
                "model": model, "topic": topic,
                "blocks": int(paired.shape[0]),
                "original_only": original_only, "flipped_only": flipped_only,
                "p_mcnemar_exact": an.mcnemar(original_only, flipped_only),
            })

        # -- topic-distribution divergence ----------------------------------
        control = responses[model_mask & (responses["persona_type"]
                                          == NO_PERSONA).to_numpy()]
        control_distribution = counts[control.index.to_numpy()].sum(axis=0) \
            if len(control) else None
        distributions = {facet: pooled_observed[i]
                         for i, facet in enumerate(facet_levels)}
        if control_distribution is not None:
            distributions[NO_PERSONA] = control_distribution
        names = list(distributions)
        for a in names:
            for b in names:
                divergence_rows.append({
                    "model": model, "row": a, "column": b,
                    "jensen_shannon": an.jensen_shannon(distributions[a],
                                                        distributions[b]),
                })

        framing_rate = fobserved[0] / np.maximum(_denominators(fobserved[0]), 1e-12)
        framing_null_rate = fnull[:, 0] / np.maximum(_denominators(fnull[:, 0]), 1e-12)
        framing_gap = framing_rate[0] - framing_rate[1]
        gap_low, gap_high = _null_band(framing_gap,
                                       framing_null_rate[:, 0] - framing_null_rate[:, 1])
        written += _persona_figures(model, figures, facet_levels, topic_names,
                                    pooled_observed, corpus, ftable, metric_rows,
                                    framing_gap, gap_low, gap_high)

    omnibus = pd.DataFrame(omnibus_rows).drop(columns=["_asymptotic_chi2"])
    omnibus["q_permutation"] = an.benjamini_hochberg(omnibus["p_permutation"])
    facet_table = pd.concat(facet_rows, ignore_index=True)
    framing_table = pd.concat(framing_rows, ignore_index=True)
    interaction = pd.DataFrame(interaction_rows)
    interaction["q_normal"] = np.nan
    testable = interaction["testable"].to_numpy()
    if testable.any():
        interaction.loc[testable, "q_normal"] = an.benjamini_hochberg(
            interaction.loc[testable, "p_normal"])
    metrics_table = pd.DataFrame(metric_rows)
    mcnemar_table = pd.DataFrame(mcnemar_rows)
    mcnemar_table["q_mcnemar"] = an.benjamini_hochberg(mcnemar_table["p_mcnemar_exact"])
    divergence = pd.DataFrame(divergence_rows)
    references = an.pairwise_reference_distances(corpus)

    sensitivity = pd.concat([
        an.detectable_difference(facet_table,
                                 group_fields=["model", "persona_type"]),
        an.detectable_difference(framing_table,
                                 group_fields=["model", "prompt_type"]),
    ], ignore_index=True)

    written += [
        _write(omnibus, out / "omnibus_tests.csv"),
        _write(sensitivity, out / "detectable_difference.csv"),
        _write(facet_table, out / "persona_type_topic_contrasts.csv"),
        _write(framing_table, out / "framing_topic_contrasts.csv"),
        _write(mcnemar_table, out / "framing_topic_mcnemar.csv"),
        _write(interaction, out / "persona_by_framing_interaction.csv"),
        _write(metrics_table, out / "response_metrics_by_condition.csv"),
        _write(divergence, out / "topic_distribution_divergence.csv"),
        _write(references, out / "reference_distances.csv"),
    ]
    written.append(out / "README.md")
    summary = {"omnibus": omnibus, "facets": facet_table,
               "framing": framing_table, "mcnemar": mcnemar_table,
               "interaction": interaction, "metrics": metrics_table,
               "divergence": divergence, "references": references,
               "sensitivity": sensitivity, "files": written}
    return summary


def _paired_indicator_tensor(indicators: np.ndarray, block: np.ndarray,
                             level: np.ndarray) -> np.ndarray:
    """(block, 2, topic) from row-wise indicators, complete blocks only."""
    n_blocks = int(block.max()) + 1
    tensor = np.full((n_blocks, 2, indicators.shape[1]), np.nan)
    tensor[block, level] = indicators
    return tensor[~np.isnan(tensor).any(axis=(1, 2))]


def _persona_figures(model, figures: Path, facet_levels, topic_names,
                     pooled_observed, corpus, framing_table, metric_rows,
                     framing_gap, gap_low, gap_high) -> list:
    slug = model.replace(".", "-")
    shown = _rank_topics(corpus)
    labels = _topic_labels(corpus)
    columns = [labels[t] for t in shown]
    keep = [topic_names.index(c) for c in columns]

    rates = pooled_observed / np.maximum(_denominators(pooled_observed), 1e-12)
    overall = pooled_observed.sum(axis=0) / max(pooled_observed.sum(), 1e-12)
    delta = (rates - overall)[:, keep] * 100

    written = viz.heatmap(
        delta, facet_levels, [_short(c) for c in columns],
        figures / f"{slug}_facet_topic_delta.png",
        title=f"{model}: topic share by disclosed facet",
        subtitle="percentage points against the model's own overall topic mix; "
                 f"largest {len(shown)} topics",
        value_label="pp vs overall", fmt="{:+.1f}")

    q = (framing_table[framing_table["prompt_type"] == ORIGINAL]
         .set_index("topic")["q_normal"])
    order = np.argsort(-np.abs(framing_gap[keep]))
    rows = [keep[i] for i in order]
    written += viz.dot_ci(
        [_short(topic_names[i], chars=34) for i in rows],
        framing_gap[rows] * 100, gap_low[rows] * 100, gap_high[rows] * 100,
        figures / f"{slug}_framing_topic_effect.png",
        highlight=[(q.get(topic_names[i]) or 1.0) < 0.05 for i in rows],
        title=f"{model}: which topics the flipped telling moves",
        subtitle="dot = observed gap; bar = central 95% of the permutation "
                 "null; colour marks BH q < 0.05, grey does not",
        xlabel="percentage points (positive = more in original_post)")

    metrics = pd.DataFrame(metric_rows)
    metrics = metrics[(metrics["model"] == model)
                      & (metrics["metric"] == "dist_to_control")]
    if len(metrics):
        pivot = (metrics.pivot(index="persona_type", columns="prompt_type",
                               values="mean")
                 .reindex(index=facet_levels, columns=[ORIGINAL, FLIPPED]))
        written += viz.grouped_bars(
            list(pivot.index),
            {framing: pivot[framing].to_numpy() for framing in pivot.columns},
            figures / f"{slug}_distance_to_control.png",
            title=f"{model}: how far each facet moves the assumptions",
            subtitle="cosine distance from the same dilemma's no-persona response",
            ylabel="cosine distance", label_fmt="{:.3f}", horizontal=True)
    return written


def _write_section_notes(out: Path, summary: dict) -> None:
    lines = ["# 1. Persona facet and story framing", ""]
    lines += [
        "Two within-subject contrasts over one shared topic space.",
        "",
        "| file | what it holds |",
        "| --- | --- |",
        "| `omnibus_tests.csv` | is there any facet / framing effect on the topic mix at all |",
        "| `persona_type_topic_contrasts.csv` | per facet x topic: share, lift, log OR, permutation p and BH q |",
        "| `framing_topic_contrasts.csv` | the same for original_post vs flipped_story |",
        "| `framing_topic_mcnemar.csv` | exact paired test on the same blocks, discordant-pair counts |",
        "| `persona_by_framing_interaction.csv` | does a facet's topic lift depend on the telling |",
        "| `response_metrics_by_condition.csv` | top-1 probability, entropy, distinct topics, distance to control |",
        "| `topic_distribution_divergence.csv` | Jensen-Shannon between every pair of facet topic mixes |",
        "| `reference_distances.csv` | what a cosine distance of this size means, from three reference contrasts |",
        "| `detectable_difference.csv` | the smallest topic-share difference each contrast had the power to find |",
        "",
        "## Reading the tests",
        "",
        "The omnibus `p_permutation` shuffles the facet label inside each "
        "(person, dilemma, framing) block and the framing label inside each "
        "(facet, person, dilemma) block. `p_asymptotic` is the ordinary "
        "chi-square on the same table and is reported next to it *only* as a "
        "diagnostic: where the two disagree by orders of magnitude, the gap is "
        "the within-subject dependence that the asymptotic test ignores.",
        "",
        "The per-cell tables carry two p-values from the same shuffles. "
        "`p_monte_carlo` is the share of shuffles at least as extreme, and "
        "cannot fall below 1/(draws + 1) -- with thousands of cells in the "
        "family that floor alone would make every corrected q-value equal 1. "
        "`p_normal` is the z against the same shuffles' mean and standard "
        "deviation, has no floor, and is what `q_normal` corrects. `testable` "
        "marks the cells where the normal approximation is safe (the null "
        "varied and the expected count is at least 5); the rest are left out "
        "of the corrected family and keep only their Monte-Carlo p.",
        "",
    ]
    omnibus = summary["omnibus"]
    lines.append("## Omnibus results")
    lines.append("")
    lines.append(_markdown_table(omnibus))
    lines.append("")
    (out / "README.md").write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# 2. the persona's own words
# ---------------------------------------------------------------------------
def section_persona_text(corpus: an.Corpus, out: Path, *,
                         persona_path: str) -> dict:
    """What is in a person's own words, against what the model assumed.

    This is what the persona side of the study can actually support. The
    demographics file that ships with the data codes each persona for age,
    gender, education, income and the rest, and the run draws 25 of the 200
    people: every one of those columns lands as two or three groups of eight,
    which is not a comparison. The transcripts are the thing the model actually
    read, and there are 250 of them.

    Two contrasts, at two grains:

    * `persona_words_by_topic.csv` -- for each topic, the transcripts are split
      at the terciles of how often the model gave that transcript that topic,
      and the two thirds' wording is contrasted. A tercile split rather than
      "did the model ever say it": each transcript draws about forty responses,
      so for a common topic almost every persona gets it at least once and a
      presence/absence contrast has no comparison group left.
    * `persona_words_by_assumption.csv` -- the repo's own
      `marked_words_by_assumption`, at the verbatim-label level, where presence
      and absence both still occur.

    Both are associations in language, not mechanisms. The transcript and the
    assumption are an input and an output of the same conditioned model.
    """
    out.mkdir(parents=True, exist_ok=True)
    figures = out / "figures"
    figures.mkdir(exist_ok=True)

    personas = an.load_persona_texts(persona_path)
    persona_words = an.persona_words_by_topic(corpus, personas)
    persona_labels = _persona_words_by_label(corpus, personas)

    written = [
        _write(persona_words, out / "persona_words_by_topic.csv"),
        _write(persona_labels, out / "persona_words_by_assumption.csv"),
    ]
    written += _persona_word_figures(figures, corpus, persona_words)
    written.append(out / "README.md")
    return {"persona_words": persona_words, "persona_labels": persona_labels,
            "files": written}


def _persona_words_by_label(corpus: an.Corpus, personas: pd.DataFrame,
                            min_count: int = 25, top: int = 8) -> pd.DataFrame:
    """The repo's own persona-text sweep, at the level of the verbatim label.

    `syco.text_analysis.marked_words_by_assumption` contrasts the transcripts
    of units where an assumption was made against matched units where it was
    not. That works at the label level, where most labels are stated by a
    minority of units; the same contrast on a topic leaves no comparison group,
    which is why `persona_words_by_topic` splits on the rate instead.
    """
    from syco.text_analysis import marked_words_by_assumption

    keys = ["persona_type", "persona_id"]
    frames = []
    for model in corpus.models:
        assumptions = corpus.assumptions[corpus.assumptions["model"] == model]
        assumptions = assumptions[assumptions["persona_id"] != NO_PERSONA]
        try:
            found = marked_words_by_assumption(
                personas, assumptions[keys + ["label"]], text_column="persona_text",
                field="label", keys=keys, min_count=min_count, top=top)
        except (ValueError, RuntimeError) as err:
            _say(f"  persona words by label skipped for {model}: {err}")
            continue
        if found.empty:
            continue
        found.insert(0, "model", model)
        frames.append(found)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _persona_word_figures(figures: Path, corpus: an.Corpus,
                          persona_words: pd.DataFrame) -> list:
    """One figure per model: the strongest persona-word signals, any topic.

    Ranked on |z| across the whole sweep rather than per topic, because the
    question is whether *anything* in the transcripts separates what the model
    assumed, and a per-topic top-10 always has a top whether or not it does.
    """
    if persona_words.empty:
        return []
    written = []
    for model in corpus.models:
        block = persona_words[persona_words["model"] == model]
        if block.empty:
            continue
        block = block.reindex(block["z"].abs().sort_values(
            ascending=False).index).head(16)
        labels = [f"{r.word} - {_short(r.topic, chars=26)}"
                  for r in block.itertuples()]
        written += viz.dot_ci(
            labels, block["z"].to_numpy(),
            np.zeros(len(block)), block["z"].to_numpy(),
            figures / f"{model.replace('.', '-')}_persona_words.png",
            highlight=_flags(block, "above_threshold").to_numpy(),
            title=f"{model}: words in a person's own description that go with "
                  "what was assumed",
            subtitle="z-scored log-odds, transcripts of the units the model "
                     "gave a topic most often against those it gave it least; "
                     "colour marks |z| >= 1.96",
            xlabel="z (positive = over-used where the topic was assumed more)")
    return written


def _write_persona_text_notes(out: Path, summary: dict) -> None:
    words = summary.get("persona_words")
    passed = (int(_flags(words, "above_threshold").sum())
              if words is not None and len(words) else 0)
    total = len(words) if words is not None else 0
    lines = [
        "# 2. The persona's own words", "",
        "What is in a person's transcript, against what the model went on to "
        "assume about them.",
        "",
        "**The demographics file is not used.** It codes each persona for age, "
        "gender, education, income and the rest, and the run draws 25 of the "
        "200 people -- every column lands as two or three groups of eight, "
        "which is not a comparison anyone should read. The transcripts are "
        "what the model actually saw and there are 250 of them, so that is "
        "what this section contrasts.",
        "",
        "| file | what it holds |",
        "| --- | --- |",
        "| `persona_words_by_topic.csv` | words over-used in the transcripts the model gave a topic to most often, against those it gave it least |",
        "| `persona_words_by_assumption.csv` | the same at the level of the verbatim assumption label |",
        "",
        f"{passed} of {total} word-and-topic pairs clear |z| = 1.96. A "
        "two-sided cut clears 5% by chance, so the count alone is not "
        "evidence; what is worth reading is whether the survivors concentrate "
        "on a few topics with wording that hangs together.",
        "",
        "Both tables are associations in language. The transcript and the "
        "assumption are an input and an output of the same conditioned model, "
        "and nothing here establishes an order between a word and a topic.",
        "",
    ]
    (out / "README.md").write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# 0. the language itself
# ---------------------------------------------------------------------------
def section_language(corpus: an.Corpus, out: Path, *, top: int = 40,
                     stopword_mode: str = "all",
                     extra_stopwords: frozenset = frozenset()) -> dict:
    """What the assumptions actually say, in words and bigrams.

    The paper's own descriptive pass -- `syco.topics.ngram_frequencies` is a
    faithful implementation of it -- run over the two things this study varies
    and nothing else: which persona type was disclosed, and which side of the
    dilemma was told.

    Three tables:

    * `term_frequencies.csv` -- what the model says most, overall and within
      each persona type, each side, and each crossing of the two.
    * `term_contrasts.csv` -- what one condition says *more* than another,
      pooled over the other factor.
    * `term_contrasts_within_side.csv` -- the same contrast with the telling
      held fixed: what a persona type adds against no persona at all, on this
      side of the story. This is what the explorer reads, because holding the
      story fixed is the only way the persona contrast means one thing.

    Function words are dropped from both levels, and so are the words the probe
    itself supplies: it addresses the person as "User A", so `user` is in 99% of
    the assumptions and half the top bigrams were `user <something>`.
    """
    out.mkdir(parents=True, exist_ok=True)
    assumptions = corpus.assumptions.copy()
    # The crossing, as its own grouping: the explorer's two dropdowns pick a
    # cell of it, and a marginal table cannot answer for a cell.
    assumptions["slice"] = (assumptions["persona_type"].astype(str) + SLICE_SEP
                            + assumptions["prompt_type"].astype(str))

    frequencies, contrasts, within = [], [], []
    for model in corpus.models:
        block = assumptions[assumptions["model"] == model]
        table = ngram_frequencies(
            block, field="description", levels=(1, 2),
            by=("persona_type", "prompt_type", "slice"), top=top, min_count=5,
            stopword_mode=stopword_mode, extra_stopwords=extra_stopwords)
        if len(table):
            table.insert(0, "model", model)
            frequencies.append(table)

        for dimension, reference in (("persona_type", NO_PERSONA),
                                     ("prompt_type", None)):
            found = an.term_contrasts(block, dimension, field="description",
                                      reference=reference, top=top,
                                      stopword_mode=stopword_mode,
                                      extra_stopwords=extra_stopwords)
            if len(found):
                found.insert(0, "model", model)
                contrasts.append(found)

        for side in (ORIGINAL, FLIPPED):
            told = block[block["prompt_type"] == side]
            if told.empty:
                continue
            found = an.term_contrasts(told, "persona_type", field="description",
                                      reference=NO_PERSONA, top=top,
                                      stopword_mode=stopword_mode,
                                      extra_stopwords=extra_stopwords)
            if len(found):
                found.insert(0, "model", model)
                found.insert(1, "prompt_type", side)
                within.append(found)
        # The control's own row: with no persona in play there is nothing to
        # contrast a persona against, so it gets the other contrast the design
        # offers -- what this side of the story says more than the other.
        control = block[block["persona_type"] == NO_PERSONA]
        if not control.empty:
            found = an.term_contrasts(control, "prompt_type",
                                      field="description", top=top,
                                      stopword_mode=stopword_mode,
                                      extra_stopwords=extra_stopwords)
            if len(found):
                found.insert(0, "model", model)
                found = found.rename(columns={"group": "prompt_type"})
                found.insert(2, "group", NO_PERSONA)
                found["dimension"] = "persona_type"
                within.append(found)

    frequency_table = (pd.concat(frequencies, ignore_index=True)
                       if frequencies else pd.DataFrame())
    contrast_table = (pd.concat(contrasts, ignore_index=True)
                      if contrasts else pd.DataFrame())
    within_table = (pd.concat(within, ignore_index=True)
                    if within else pd.DataFrame())
    written = [
        _write(frequency_table, out / "term_frequencies.csv"),
        _write(contrast_table, out / "term_contrasts.csv"),
        _write(within_table, out / "term_contrasts_within_side.csv"),
        out / "README.md",
    ]
    return {"frequencies": frequency_table, "contrasts": contrast_table,
            "within": within_table, "files": written}


def _write_language_notes(out: Path, summary: dict) -> None:
    frequencies = summary.get("frequencies")
    lines = [
        "# 0. What the assumptions say", "",
        "The paper's own descriptive pass -- unigram and bigram frequency over "
        "the assumption explanations -- run over the two things this study "
        "varies: the persona type disclosed, and the side of the dilemma told. "
        "`syco.topics.ngram_frequencies` is the implementation.",
        "",
        "Function words are dropped from both levels (`--stopwords unigrams` "
        "restores the paper's setting, which keeps them in bigrams), and so "
        "are the words the probe itself supplies. It addresses the person as "
        "\"User A\", so `user` sits in 99% of the assumptions and half the top "
        "bigrams were `user <something>`; `--keep-prompt-echo` leaves them in.",
        "",
        "| file | what it holds |",
        "| --- | --- |",
        "| `term_frequencies.csv` | how often each word and bigram appears, overall and within each persona type, each side, and each crossing |",
        "| `term_contrasts.csv` | which terms one condition uses *more* than another, pooled over the other factor |",
        "| `term_contrasts_within_side.csv` | the same with the telling held fixed -- what a persona type adds against no persona, on this side |",
        "",
        "**Read a contrast table, not the frequency table, when comparing "
        "conditions.** The top of the frequency table is what every condition "
        "shares, which describes the corpus accurately and says nothing about "
        "which setting produced it.",
        "",
    ]
    if frequencies is not None and len(frequencies):
        overall = frequencies[frequencies["scope"] == "overall"]
        for level in ("unigram", "2-gram"):
            block = overall[overall["level"] == level]
            if block.empty:
                continue
            lines += [f"## Most common {level}s, by model", ""]
            for model, rows in block.groupby("model"):
                terms = ", ".join(f"`{r.term}` {r.share_assumptions:.0%}"
                                  for r in rows.head(12).itertuples())
                lines += [f"- **{model}**: {terms}", ""]
    (out / "README.md").write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# 3. sycophancy
# ---------------------------------------------------------------------------
def section_sycophancy(corpus: an.Corpus, score_sources: dict, out: Path, *,
                       n_perm: int, seed: int, min_count: int) -> dict:
    """Which assumptions travel with the forced-choice sycophancy score.

    Sycophancy here is Neplenbroek et al.'s: absolving the flipped-story teller
    too, given that the model absolved the original poster, read off the first
    generated token's Yes/No log-probabilities. It is a property of a
    (facet, person, dilemma) cell.

    Two things about this join have to stay in view.

    **Every join here is within-model.** A model is analyzed only against its
    own forced-choice collection; a model with no collection of its own is
    skipped and named in `sycophancy_coverage.csv` rather than being scored
    against somebody else's numbers. Joined across models the result would not
    be "this model's assumptions go with this model's sycophancy" but "these
    assumptions were verbalized on cells that *some* model finds easy to be
    sycophantic on", which is a statement about the items and reads as if it
    were a statement about the model.

    **The dilemma dominates.** Sycophancy varies far more between dilemmas than
    between personas, so every contrast is taken inside a dilemma and pooled.
    The raw mean is on the tables too, and where the two disagree the raw one is
    reporting which dilemma the assumption was stated on.
    """
    out.mkdir(parents=True, exist_ok=True)
    figures = out / "figures"
    figures.mkdir(exist_ok=True)

    responses = corpus.responses.reset_index(drop=True)
    counts = an.topic_counts(corpus)
    labels = _topic_labels(corpus)
    topic_names = [labels[int(t)] for t in corpus.topic_ids]
    # The merge below is a left join on a unique key, so it preserves order --
    # but the topic-count matrix is indexed by position and a silent reorder
    # would misalign every row, so the position travels as a column.
    responses = responses.assign(_row=np.arange(len(responses)))

    keys = ["persona_type", "persona_id", "prompt_id"]
    coverage_rows, tercile_frames, label_frames, logit_rows = [], [], [], []
    written, skipped = [], []

    for model in corpus.models:
        if model not in score_sources:
            skipped.append(model)
            coverage_rows.append({
                "assumptions_model": model, "sycophancy_model": None,
                "framing": None, "role": "skipped -- no forced-choice "
                "collection for this model", "responses": int(
                    (corpus.responses["model"] == model).sum()),
                "matched_to_a_score": 0, "match_rate": 0.0})
    if skipped:
        _say("  no forced-choice collection for: " + ", ".join(skipped)
             + " -- skipped rather than scored against another model's")
    for model, (score_model, scores) in sorted(score_sources.items()):
        scores = _prepare_scores(scores)
        lean = scores[keys + ["logit_no_flipped_story", "eligible",
                              "sycophancy", "sycophancy_soft",
                              "sycophancy_logit"]].copy()
        lean["sycophancy"] = lean["sycophancy"].astype("Float64").astype("float64")
        joined = responses[responses["model"] == model].merge(
            lean, on=keys, how="left")

        for framing, framing_label in ((FLIPPED, "flipped_story (primary)"),
                                       (ORIGINAL, "original_post (sensitivity)")):
            block = joined[joined["prompt_type"] == framing]
            scored = block.dropna(subset=["sycophancy"])
            coverage_rows.append({
                "assumptions_model": model, "sycophancy_model": score_model,
                "framing": framing, "role": framing_label.split(" ")[-1].strip("()"),
                "responses": len(block), "matched_to_a_score": len(scored),
                "match_rate": len(scored) / max(len(block), 1),
                "mean_sycophancy": float(scored["sycophancy"].mean())
                if len(scored) else float("nan"),
                "mean_sycophancy_soft": float(scored["sycophancy_soft"].mean())
                if len(scored) else float("nan"),
                "dilemmas": int(scored["prompt_id"].nunique()),
                # The 0/1 indicator is constant inside most dilemmas, so a
                # within-dilemma contrast on it has almost nothing to work
                # with. The continuous log-odds behind it varies inside all of
                # them, which is why it is the primary outcome below.
                "dilemmas_binary_varies": int(
                    scored.groupby("prompt_id")["sycophancy"].nunique().gt(1).sum()),
                "dilemmas_logit_varies": int(
                    scored.groupby("prompt_id")["logit_no_flipped_story"]
                    .nunique().gt(1).sum()),
                "logit_sd_within_dilemma": float(
                    scored.groupby("prompt_id")["logit_no_flipped_story"]
                    .std().mean()) if len(scored) else float("nan"),
            })

        primary = joined[joined["prompt_type"] == FLIPPED].dropna(
            subset=["logit_no_flipped_story"])
        if primary.empty:
            continue

        # -- most vs least sycophantic, terciles taken inside each dilemma ---
        tercile = _within_dilemma_terciles(primary).reindex(primary.index)
        keep = tercile.notna().to_numpy()
        rows = primary[keep]
        row_positions = rows["_row"].to_numpy()
        stratum, _ = _codes(rows["prompt_id"])
        level, level_names = _ordered_codes(
            tercile[keep], ["least sycophantic", "middle", "most sycophantic"])
        observed, null, level_counts = an.stratified_permutation(
            counts[row_positions], stratum, level, n_perm=n_perm, seed=seed)
        report = an.omnibus_from_permutation(observed, null)
        rate_null = null / np.maximum(_denominators(null), 1e-12)
        rate_observed = observed / np.maximum(_denominators(observed), 1e-12)
        gap = rate_observed[2] - rate_observed[0]
        gap_low, gap_high = _null_band(gap, rate_null[:, 2] - rate_null[:, 0])
        contrast = an.contrast_table(
            observed, null, _denominators(observed), level_names, topic_names,
            level_field="tercile", feature_field="topic")
        contrast.insert(0, "assumptions_model", model)
        contrast.insert(1, "sycophancy_model", score_model)
        contrast["omnibus_p_permutation"] = report["p_permutation"]
        tercile_frames.append(contrast)

        # -- one regression per topic, dilemma fixed effects -----------------
        logit_rows += _per_topic_effect(rows, counts[row_positions], topic_names,
                                        model, score_model)

        # -- the repo's own label-level ranking, on both outcomes ------------
        for outcome in ("sycophancy_logit", "sycophancy"):
            label_frames.append(_assumption_ranking(
                corpus, rows, scores, model, score_model, min_count, outcome))

        written += _sycophancy_figures(figures, corpus, model, contrast,
                                       level_names, topic_names, gap,
                                       gap_low, gap_high)

    coverage = pd.DataFrame(coverage_rows)
    terciles = (pd.concat(tercile_frames, ignore_index=True)
                if tercile_frames else pd.DataFrame())
    ranking = (pd.concat(label_frames, ignore_index=True)
               if label_frames else pd.DataFrame())
    logits = pd.DataFrame(logit_rows)
    if len(logits):
        logits["q_value"] = an.benjamini_hochberg(logits["p_value"])
        logits = logits.sort_values(["assumptions_model", "p_value"])

    sensitivity = (an.detectable_difference(
        terciles, group_fields=["assumptions_model", "tercile"])
        if len(terciles) else pd.DataFrame())

    written = list(written) + [
        _write(coverage, out / "sycophancy_coverage.csv"),
        _write(sensitivity, out / "detectable_difference.csv"),
        _write(terciles, out / "topics_most_vs_least_sycophantic.csv"),
        _write(ranking, out / "assumptions_ranked_by_sycophancy.csv"),
        _write(logits, out / "topic_logit_within_dilemma.csv"),
    ]
    written.append(out / "README.md")
    summary = {"coverage": coverage, "terciles": terciles, "ranking": ranking,
               "logits": logits, "sensitivity": sensitivity, "files": written,
               "skipped": skipped}
    return summary


def _prepare_scores(scores: pd.DataFrame) -> pd.DataFrame:
    """Add the log-odds column the contrasts are taken on.

    The probability saturates: for most cells it is 1 - 1e-8 or 1e-8, so within
    a dilemma its variance is floating-point noise and the pooled standard
    error behind any contrast on it collapses, turning a difference of a
    millionth into p = 1e-9. The log-odds behind it has a within-dilemma spread
    of over one unit, and is the scale everything here uses.
    """
    scores = scores.copy()
    scores["sycophancy_logit"] = scores["logit_no_flipped_story"].where(
        scores["eligible"])
    return scores


def _within_dilemma_terciles(rows: pd.DataFrame) -> pd.Series:
    """Least / middle / most sycophantic, ranked inside each dilemma.

    Inside, because between dilemmas the score is mostly the dilemma: some are
    absolved whatever the model thinks it is talking to, some never are. A
    global tercile split would produce a "most sycophantic" group that is
    really a list of the easy dilemmas, and the topics separating it would be
    the topics of those stories.

    The continuous log-odds is ranked rather than the 0/1 indicator, which is
    constant inside many dilemmas and would leave nothing to split.
    """
    def cut(series):
        if series.nunique() < 3:
            return pd.Series(pd.NA, index=series.index, dtype="object")
        ranked = series.rank(method="first", pct=True)
        return pd.Series(
            np.where(ranked <= 1 / 3, "least sycophantic",
                     np.where(ranked > 2 / 3, "most sycophantic", "middle")),
            index=series.index, dtype="object")

    return (rows.groupby("prompt_id", group_keys=False)["logit_no_flipped_story"]
            .apply(cut))


def _per_topic_effect(rows: pd.DataFrame, counts: np.ndarray, topic_names: list,
                      model: str, score_model: str,
                      min_responses: int = 40) -> list:
    """`log-odds(absolve the flipped teller) ~ topic + dilemma`, one fit per topic.

    The outcome is the **continuous** score, not the 0/1 indicator. The
    indicator is constant inside all but a handful of dilemmas, so a
    within-dilemma model of it would be estimated off those few and would say
    nothing about the rest; the log-odds behind it varies inside every dilemma
    and keeps all twenty in the estimate.

    One topic at a time rather than all at once: three hundred correlated
    dummies over 25 people identifies nothing. Standard errors are clustered on
    the person, which is the level at which responses repeat. Twenty-five
    clusters is few, so this is a cross-check on the within-dilemma permutation
    tests, not a replacement for them.
    """
    import statsmodels.api as sm

    outcome = rows["logit_no_flipped_story"].to_numpy("float64")
    usable = np.isfinite(outcome)
    if usable.sum() < min_responses or np.ptp(outcome[usable]) == 0:
        return []
    dilemmas = pd.get_dummies(rows.loc[usable, "prompt_id"], drop_first=True,
                              dtype="float64").to_numpy()
    clusters = pd.Categorical(rows["persona_id"]).codes[usable]

    out = []
    for j, topic in enumerate(topic_names):
        present = (counts[:, j] > 0).astype("float64")[usable]
        if present.sum() < min_responses or present.sum() == len(present):
            continue
        design = np.column_stack([np.ones(len(present)), present, dilemmas])
        try:
            fit = sm.OLS(outcome[usable], design).fit(
                cov_type="cluster", cov_kwds={"groups": clusters})
        except Exception:
            continue
        out.append({
            "assumptions_model": model, "sycophancy_model": score_model,
            "topic": topic, "n_responses": int(usable.sum()),
            "n_mentioning": int(present.sum()),
            "delta_logit": float(fit.params[1]),
            "se_clustered": float(fit.bse[1]),
            "t": float(fit.tvalues[1]), "p_value": float(fit.pvalues[1]),
            "n_clusters": len(np.unique(clusters)),
        })
    return out


def _assumption_ranking(corpus: an.Corpus, rows: pd.DataFrame,
                        scores: pd.DataFrame, model: str, score_model: str,
                        min_count: int, outcome: str) -> pd.DataFrame:
    """`syco.sycophancy.sycophancy_by_assumption` on the free-text labels.

    The repo's own implementation, at the level the paper reads: the assumption
    as stated, not the cluster it fell into. `within_delta` is the contrast
    taken inside a dilemma and is the column to read; the raw mean beside it is
    confounded by the dilemma.

    Run on two outcomes. `sycophancy_logit` is the log-odds of absolving the
    flipped teller and varies inside every dilemma; `sycophancy` is the 0/1
    indicator, which in all but four dilemmas takes the same value in every
    cell and so contributes a contrast of exactly zero by construction.
    Reading only the second would report a null that belongs to the threshold
    rather than to the assumptions.

    The probability `sycophancy_soft` is deliberately not one of them. It is
    the same quantity as the logit through a sigmoid, but saturated: most cells
    sit within 1e-8 of 0 or 1, so the within-dilemma variance is
    floating-point noise, the pooled standard error collapses, and differences
    of a millionth come back at p = 1e-9. That is a property of the scale.
    """
    keys = [k for k in an.RESPONSE_KEYS if k in corpus.assumptions.columns]
    subset = corpus.assumptions.merge(rows[keys], on=keys, how="inner")
    joined, _ = syc.attach_to_assumptions(subset, scores, level="cell")
    table = syc.sycophancy_by_assumption(joined, field="label",
                                         min_count=min_count, column=outcome,
                                         within="prompt_id")
    # The helper names its mean column "sycophancy" whichever outcome it was
    # given; renamed here so the two runs stack into one file without one
    # outcome's numbers silently sitting in the other's column.
    table = table.rename(columns={"sycophancy": f"mean_{outcome}"})
    table.insert(0, "assumptions_model", model)
    table.insert(1, "sycophancy_model", score_model)
    table.insert(2, "outcome", outcome)
    return table


def _sycophancy_figures(figures: Path, corpus: an.Corpus, model: str,
                        contrast: pd.DataFrame, level_names: list,
                        topic_names: list, gap, gap_low, gap_high) -> list:
    slug = model.replace(".", "-")
    shown = _rank_topics(corpus, TOP_TOPICS)
    labels = _topic_labels(corpus)
    columns = [labels[t] for t in shown]
    ends = [level_names[0], level_names[-1]]
    pivot = (contrast[contrast["topic"].isin(columns)]
             .pivot(index="tercile", columns="topic", values="rate")
             .reindex(index=ends, columns=columns))
    written = viz.grouped_bars(
        [_short(c, chars=26) for c in pivot.columns],
        {level: pivot.loc[level].to_numpy() * 100 for level in ends},
        figures / f"{slug}_topics_by_sycophancy_tercile.png",
        title=f"{model}: topic mix at the two ends of the sycophancy scale",
        subtitle="share of verbalized assumptions; terciles taken inside each "
                 "dilemma, sycophancy scored on the forced-choice collection",
        ylabel="% of assumptions", label_fmt="", horizontal=True,
        figsize=(7.4, 0.30 * len(columns) * 2 + 1.8))

    q = (contrast[contrast["tercile"] == level_names[-1]]
         .set_index("topic")["q_normal"])
    keep = [topic_names.index(c) for c in columns]
    order = [keep[i] for i in np.argsort(gap[keep])]
    written += viz.dot_ci(
        [_short(topic_names[i], chars=34) for i in order],
        gap[order] * 100, gap_low[order] * 100, gap_high[order] * 100,
        figures / f"{slug}_sycophancy_topic_gap.png",
        highlight=[(q.get(topic_names[i]) or 1.0) < 0.05 for i in order],
        title=f"{model}: topics that separate the sycophancy terciles",
        subtitle="most minus least sycophantic; dot = observed, bar = central "
                 "95% of the within-dilemma permutation null",
        xlabel="percentage points (positive = more where sycophancy is higher)")
    return written


def _write_sycophancy_notes(out: Path, summary: dict, corpus) -> None:
    coverage = summary["coverage"]
    scored = coverage[coverage["sycophancy_model"].notna()] if len(coverage) \
        else coverage
    pairs = (scored[["assumptions_model", "sycophancy_model"]]
             .drop_duplicates().itertuples(index=False)) if len(scored) else []
    lines = [
        "# 3. Sycophancy", "",
        "Sycophancy is absolving the flipped-story teller too, given that the "
        "model absolved the original poster, read off the first generated "
        "token's constrained Yes/No log-probabilities.",
        "",
        "**Every join here is within-model.** A model is scored only against "
        "its own forced-choice collection.",
        "",
    ]
    if pairs:
        lines += ["| assumptions | scored against |", "| --- | --- |"]
        lines += [f"| `{row.assumptions_model}` | `{row.sycophancy_model}` |"
                  for row in pairs]
        lines.append("")
    skipped = summary.get("skipped") or []
    if skipped:
        lines += [
            "Skipped, having no forced-choice collection of their own: "
            + ", ".join(f"`{model}`" for model in skipped)
            + ". Scoring them against another model's collection would answer "
            "a different question -- which (persona, dilemma) cells invite the "
            "behaviour -- while reading as if it were about the responding "
            "model, so it is not done here. Collect "
            "`<model>_results.pkl` for them and re-run to fill this in.",
            "",
        ]
    if not len(scored):
        lines += [
            "**No model in this run has a collection of its own, so this "
            "section is empty.** The tables below exist and are headed but "
            "hold no rows.",
            "",
        ]
    lines += [
        "Terciles are taken **inside each dilemma** on the continuous "
        "log-odds. A global split would sort dilemmas, not personas: some "
        "stories are absolved whatever the model thinks it is talking to.",
        "",
        "Two outcomes are reported side by side. `sycophancy_logit` is the "
        "log-odds of absolving the flipped teller and varies inside every "
        "dilemma; `sycophancy` is the reference implementation's 0/1 "
        "indicator, which in most dilemmas takes one value in every cell and "
        "so contributes a contrast of exactly zero by construction. The "
        "probability `sycophancy_soft` is deliberately absent: it saturates "
        "within 1e-8 of 0 or 1, so its within-dilemma variance is "
        "floating-point noise and differences of a millionth come back at "
        "p = 1e-9.",
        "",
        "| file | what it holds |",
        "| --- | --- |",
        "| `sycophancy_coverage.csv` | which model was scored against what, how much of the grid carries a score, and the rate |",
        "| `topics_most_vs_least_sycophantic.csv` | topic share per tercile, with the within-dilemma permutation test |",
        "| `assumptions_ranked_by_sycophancy.csv` | free-text labels ranked by within-dilemma delta, on both outcomes (`outcome` column) |",
        "| `topic_logit_within_dilemma.csv` | per-topic regression on the log-odds, dilemma fixed effects, person-clustered SE |",
        "| `detectable_difference.csv` | the smallest topic-share gap between terciles the test had the power to find |",
        "",
    ]
    if len(coverage):
        lines += ["## Coverage", "", _markdown_table(coverage), ""]
    (out / "README.md").write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------
def _shared_tables(corpus: an.Corpus, out: Path) -> list:
    shared = out / "shared"
    shared.mkdir(parents=True, exist_ok=True)
    topics = corpus.topics.copy()
    written = [_write(topics, shared / "topics.csv")]

    responses = corpus.responses.drop(columns=["text"], errors="ignore").copy()
    # The response-level embedding matrix rides on `.attrs`; parquet tries to
    # JSON-encode that and cannot. The vectors belong in the cache, not here.
    responses.attrs = {}
    responses["topics"] = responses["topics"].map(
        lambda t: " ".join(str(x) for x in t))
    responses["dist_to_control"] = an.distance_to_control(corpus).to_numpy()
    responses.to_parquet(shared / "responses.parquet", index=False)
    written.append(shared / "responses.parquet")

    keys = [k for k in an.RESPONSE_KEYS if k in corpus.assumptions.columns]
    corpus.assumptions[keys + ["rank", "assumption", "label", "description",
                               "probability_norm", "topic"]].to_parquet(
        shared / "assumption_topics.parquet", index=False)
    written.append(shared / "assumption_topics.parquet")

    profile = pd.DataFrame([{
        "model": model,
        "responses": int((corpus.responses["model"] == model).sum()),
        "assumptions": int((corpus.assumptions["model"] == model).sum()),
        "personas": int(corpus.responses.loc[
            corpus.responses["model"] == model, "persona_id"].nunique()) - 1,
        "facets": int(corpus.responses.loc[
            corpus.responses["model"] == model, "persona_type"].nunique()) - 1,
        "dilemmas": int(corpus.responses.loc[
            corpus.responses["model"] == model, "prompt_id"].nunique()),
        "distinct_labels": int(corpus.assumptions.loc[
            corpus.assumptions["model"] == model, "label"].nunique()),
        "outlier_share": float((corpus.assumptions.loc[
            corpus.assumptions["model"] == model, "topic"] < 0).mean()),
    } for model in corpus.models])
    written.append(_write(profile, shared / "corpus_profile.csv"))
    return written


def _key_findings(corpus, one: dict, two: dict, three: dict,
                  out: Path, meta: dict) -> Path:
    """The numbers that survived their own test, and the ones that did not.

    Written so a null result is as visible as a positive one -- with 25 people
    behind the demographic tables, "no detectable difference" is the most
    likely honest answer for most columns and it has to be stated rather than
    left to be inferred from an absence.
    """
    L = ["# Key findings", "",
         f"Generated {meta['generated']} from {len(corpus.models)} model(s): "
         f"{', '.join(f'`{m}`' for m in corpus.models)}.",
         "",
         f"{meta['n_responses']:,} responses carrying {meta['n_assumptions']:,} "
         f"verbalized assumptions, clustered into "
         f"{corpus.n_topics} topics "
         f"({corpus.outlier_share:.1%} left unclustered). "
         f"All tests use {meta['n_perm']:,} permutations.",
         ""]

    # -- 1 -----------------------------------------------------------------
    L += ["## 1. Persona facet and story framing", ""]
    omnibus = one["omnibus"]
    for _, row in omnibus[omnibus["framing"] == "both"].iterrows():
        verdict = ("changes the topic mix" if row["q_permutation"] < 0.05
                   else "does **not** detectably change the topic mix")
        L.append(f"- **{row['model']} / {row['factor']}** {verdict}: "
                 f"Cramer's V = {row['cramers_v']:.3f}, permutation "
                 f"p = {_fmt_p(row['p_permutation'])} "
                 f"(asymptotic chi-square would say {_fmt_p(row['p_asymptotic'])}).")
    L.append("")

    by_model = (one["facets"][one["facets"]["q_normal"] < 0.05]
                .groupby("model").size().reindex(corpus.models).fillna(0)
                .astype(int))
    L.append("- Per-topic, that aggregate effect is carried very unevenly: "
             + ", ".join(f"**{model}** {n} facet x topic cells survive "
                         "correction" for model, n in by_model.items())
             + ". A model with a significant omnibus and no surviving cell has "
               "the same size of effect spread thin across topics rather than "
               "concentrated in any of them.")
    L.append("")

    references = one["references"]
    distances = one["metrics"][one["metrics"]["metric"] == "dist_to_control"]
    for model in corpus.models:
        block = references[references["model"] == model].set_index("comparison")
        mine = (distances[distances["model"] == model]
                .groupby("persona_type", as_index=False)["mean"].mean())
        if mine.empty:
            continue
        top = mine.sort_values("mean").iloc[[0, -1]]
        facet_gap = block.loc["same_person_other_facet", "mean_distance"]
        person_gap = block.loc["other_person_same_facet", "mean_distance"]
        telling_gap = block.loc["cross_framing", "mean_distance"]
        L.append(
            f"- **{model}** -- the facet that moves the assumptions least is "
            f"`{top.iloc[0]['persona_type']}` "
            f"({top.iloc[0]['mean']:.3f} cosine from the no-persona answer) and "
            f"most is `{top.iloc[-1]['persona_type']}` ({top.iloc[-1]['mean']:.3f}). "
            "Against three references measured on the same vectors: changing "
            f"which facet is disclosed moves the picture {facet_gap:.3f}, "
            f"changing which person entirely moves it {person_gap:.3f}, and "
            f"changing only the telling moves it {telling_gap:.3f}. "
            "**Which facet of someone is on the table moves the model's read "
            "of them about as far as swapping in a different person "
            f"({facet_gap:.3f} vs {person_gap:.3f}); retelling the same events "
            "from the other side moves it further than either.**")
    L.append("")

    framing = one["framing"]
    for model in corpus.models:
        moved = framing[(framing["model"] == model)
                        & (framing["prompt_type"] == ORIGINAL)
                        & (framing["q_normal"] < 0.05)]
        if moved.empty:
            L.append(f"- **{model}** -- no topic's share differs between the two "
                     "tellings after correction.")
            continue
        moved = moved.reindex(moved["delta"].abs().sort_values(ascending=False).index)
        head = moved.head(3)
        L.append(f"- **{model}** -- {len(moved)} topics shift between tellings "
                 "(BH q < 0.05). Largest: " + "; ".join(
                     f"`{r.topic}` {r.delta:+.1%} toward "
                     f"{'original_post' if r.delta > 0 else 'flipped_story'}"
                     for r in head.itertuples()) + ".")
    L.append("")

    metrics = one["metrics"]
    for metric, description in (
            ("top1_prob", "confidence in its leading mental model"),
            ("n_topics", "distinct topics per response")):
        block = metrics[metrics["metric"] == metric]
        if block.empty:
            continue
        spread = (block.groupby(["model", "persona_type"])["mean"].mean()
                  .groupby(level=0).agg(["min", "max", "idxmin", "idxmax"]))
        for model, row in spread.iterrows():
            L.append(f"- **{model}** -- {description} ranges from "
                     f"{row['min']:.3f} (`{row['idxmin'][1]}`) to "
                     f"{row['max']:.3f} (`{row['idxmax'][1]}`) across facets.")
    L.append("")

    interaction = one["interaction"]
    survivors = interaction[interaction["q_normal"] < 0.05]
    testable = int(interaction["testable"].sum())
    if corpus.n_topics:
        L.append(f"- **Read the framing numbers as an upper bound.** "
                 f"{corpus.story_bound:.0%} of the {corpus.n_topics} topics "
                 f"sit in one story or a handful (median spread across "
                 f"dilemmas {corpus.median_spread:.2f} of a possible 1.00), "
                 "because they were clustered from the explanations and an "
                 "explanation restates the story, so flipping the telling "
                 "changes the wording almost by definition. The facet contrast "
                 "is taken inside a dilemma so it is not inflated the same "
                 "way, but it is narrowed to 'which of this story's readings' "
                 "rather than 'which reading of a person'. `--field "
                 "assumption` clusters the short labels instead, which cut "
                 "across dilemmas.")
    L.append("")
    L.append(f"- Facet-by-framing interaction: {len(survivors)} of the "
             f"{testable} testable facet x topic cells "
             f"({testable / max(len(interaction), 1):.0%} of the grid -- the "
             "rest have too few assumptions on one side of the telling to "
             "compare) have a lift that depends on the telling (BH q < 0.05). "
             + ("Within what is testable, the facet effect is the same under "
                "both tellings."
                if survivors.empty else
                "See `01_persona_framing/persona_by_framing_interaction.csv`."))
    L.append("")

    # -- 2 -----------------------------------------------------------------
    L += ["## 2. The persona's own words", "",
          "The demographics file that ships with the data is not used. It "
          "codes each persona for age, gender, education, income and the "
          "rest, and the run draws 25 of the 200 people: every column lands "
          "as two or three groups of eight, which is not a comparison. The "
          "transcripts are what the model actually read, and there are 250 "
          "of them.", ""]
    persona_words = two.get("persona_words")
    if persona_words is not None and len(persona_words):
        cleared = persona_words[_flags(persona_words, "above_threshold")]
        L.append(f"- {len(cleared)} of {len(persona_words)} "
                 f"({len(cleared) / len(persona_words):.0%}) word-and-topic "
                 "pairs clear |z| = 1.96 in the contrast between the "
                 "transcripts the model gave a topic to most often and those "
                 "it gave it least. A two-sided cut clears 5% by chance, so "
                 "the count alone is not evidence; the reading is in whether "
                 f"the survivors concentrate -- they land on "
                 f"{cleared['topic'].nunique()} of the "
                 f"{persona_words['topic'].nunique()} topics tested.")
        for model, block in cleared.groupby("model"):
            block = block.reindex(block["z"].abs().sort_values(
                ascending=False).index).head(5)
            L.append(f"  - **{model}**: " + "; ".join(
                f"`{r.word}` {r.z:+.1f} with `{r.topic}`"
                for r in block.itertuples()) + ".")
    persona_labels = two.get("persona_labels")
    if persona_labels is not None and len(persona_labels):
        L.append(f"- At the level of the verbatim assumption, "
                 f"{len(persona_labels)} persona-word associations passed the "
                 "repo's own marked-word threshold "
                 "(`persona_words_by_assumption.csv`).")
    L.append("")

    # -- 3 -----------------------------------------------------------------
    L += ["## 3. Sycophancy", ""]
    coverage = three["coverage"]
    skipped = (coverage[coverage["sycophancy_model"].isna()]["assumptions_model"]
               .tolist() if len(coverage) else [])
    if skipped:
        L.append("- Skipped for want of a forced-choice collection of their "
                 "own: " + ", ".join(f"`{m}`" for m in skipped)
                 + ". Scoring them against another model's collection would "
                 "measure which cells invite the behaviour, not what the "
                 "model does, so it is not done.")
    primary = coverage[(coverage["framing"] == FLIPPED)
                       & coverage["sycophancy_model"].notna()] \
        if len(coverage) else coverage
    if not len(primary):
        L.append("- No model in this run has a collection of its own, so the "
                 "section is empty.")
    for _, row in primary.iterrows():
        L.append(f"- **{row['assumptions_model']}** against its own "
                 f"`{row['sycophancy_model']}` collection: "
                 f"{row['match_rate']:.1%} of flipped-story responses carry a "
                 f"score, mean sycophancy {row['mean_sycophancy']:.3f} over "
                 f"{int(row['dilemmas'])} dilemmas. The 0/1 indicator varies "
                 f"inside only {int(row['dilemmas_binary_varies'])} of them; "
                 f"the continuous log-odds varies inside "
                 f"{int(row['dilemmas_logit_varies'])}.")
    terciles = three["terciles"]
    if len(terciles):
        for model in corpus.models:
            block = terciles[terciles["assumptions_model"] == model]
            if block.empty:
                continue
            top = block[block["tercile"] == "most sycophantic"]
            hits = top[top["q_normal"] < 0.05]
            hits = hits.reindex(hits["delta"].abs().sort_values(
                ascending=False).index)
            if hits.empty:
                L.append(f"- **{model}** -- no topic separates the most from "
                         "the least sycophantic third after correction.")
            else:
                L.append(f"- **{model}** -- {len(hits)} "
                         f"topic{'' if len(hits) == 1 else 's'} over- or "
                         "under-used where sycophancy is highest "
                         "(BH q < 0.05). Largest: " + "; ".join(
                             f"`{r.topic}` {r.delta:+.1%}"
                             for r in hits.head(4).itertuples()) + ".")
    ranking = three["ranking"]
    if len(ranking):
        for outcome in ranking["outcome"].unique():
            block = ranking[ranking["outcome"] == outcome]
            survivors = block[block["q_value"] < 0.05]
            L.append(f"- Verbatim assumption labels against `{outcome}`: "
                     f"{len(survivors)} of {len(block)} labels stated often "
                     "enough to rank have a within-dilemma delta that survives "
                     "BH correction.")
            if len(survivors):
                top = survivors.reindex(
                    survivors["within_delta"].abs().sort_values(
                        ascending=False).index).head(4)
                L.append("  - Largest: " + "; ".join(
                    f"{r.assumptions_model} `{r.label}` "
                    f"{r.within_delta:+.3f} (n={int(r.n)})"
                    for r in top.itertuples()) + ".")

    L.append("")
    L += ["## What these tables cannot say", "",
          "- Every contrast is an association inside one model's own outputs. "
          "The assumption and the behavior are two readings of the same "
          "conditioned model, so a topic that travels with sycophancy has not "
          "been shown to cause it.",
          "- A topic is a cluster of wordings. That it separates two groups is "
          "a fact about the model's language, not about the groups.",
          "- The persona-text contrasts are associations in language. A "
          "transcript and an assumption are an input and an output of the "
          "same conditioned model; no order between them is established.",
          f"- The {len(corpus.models)} models here were served as 4-bit GGUF "
          "quantizations at temperature 0.7 with one draw per cell. Nothing "
          "here separates a model's behaviour from its serving configuration "
          "or from sampling noise in a single draw.",
          ""]
    path = out / "KEY_FINDINGS.md"
    path.write_text("\n".join(L))
    return path


def _index(out: Path, corpus, meta: dict) -> Path:
    """The directory's own README, listing what is actually on disk.

    Walked rather than accumulated as the run goes, so it stays right after a
    `--sections report` rebuild and after a partial run, and can never claim a
    file that was not written.
    """
    sections = {
        "0. Words and bigrams": "00_language",
        "1. Persona facet and framing": "01_persona_framing",
        "2. The persona's own words": "02_persona_text",
        "3. Sycophancy": "03_sycophancy",
        "Shared": "shared",
    }
    lines = [
        "# Open-ended assumptions: analysis output", "",
        f"Generated {meta['generated']} by `python -m syco analyze`.", "",
        "```", meta["command"], "```", "",
        "## Start here", "",
        "- [KEY_FINDINGS.md](KEY_FINDINGS.md) -- what survived its own test.",
        "- [report.html](report.html) -- the same thing as one self-contained "
        "page, figures and tables included.",
        "- [01_persona_framing/](01_persona_framing/README.md) -- facet and framing.",
        "- [02_persona_text/](02_persona_text/README.md) -- the persona's own words against what was assumed.",
        "- [03_sycophancy/](03_sycophancy/README.md) -- assumptions against the sycophancy score.",
        "",
        "## Inputs", "",
        "| model | assumptions table |", "| --- | --- |",
    ]
    lines += [f"| `{alias}` | `{path}` |"
              for alias, path in meta.get("sources", {}).items()]
    lines += [
        "",
        f"- persona transcripts: `{meta.get('personas')}`",
        "- forced-choice collections, each model against its own: "
        + (", ".join(f"`{model}` -> `{path}`"
                     for model, path in (meta.get("scores") or {}).items())
           or "none found -- section 3 is empty"),
        "",
        "## Shared", "",
        "| file | what it holds |", "| --- | --- |",
        "| `shared/topics.csv` | the topic space: size, top words, examples, and how evenly each topic spreads across the dilemmas |",
        "| `shared/responses.parquet` | one row per probe completion, with its topic set and metrics |",
        "| `shared/assumption_topics.parquet` | one row per assumption, with its topic |",
        "| `shared/corpus_profile.csv` | per-model counts, distinct labels, outlier share |",
        "| `shared/cache/` | embeddings and the fitted topic model, reused on re-run |",
        "",
        "## Method in one paragraph", "",
        "One sentence-transformer topic space is fitted over every model's "
        "assumptions pooled, so a topic means the same thing everywhere. Each "
        "response contributes its k assumptions' topic counts. The three "
        "sections then use two different nulls, because the design gives "
        "each question a different unit of independence: persona type and the "
        "side of the story are shuffled inside each person-and-dilemma block; "
        "sycophancy inside each dilemma. Every table carries the "
        "asymptotic chi-square p as well, so the gap between it and the "
        "permutation p stays visible.",
        "",
        "Every prose file here is rebuilt from the tables, so it can be "
        "regenerated without repeating the permutations:",
        "",
        "```bash",
        f"python -m syco analyze --sections report --out {out}",
        "```",
        "",
        "## Files written", "",
    ]
    for name, folder in sections.items():
        directory = out / folder
        if not directory.exists():
            continue
        produced = sorted(path for path in directory.rglob("*")
                          if path.is_file() and "cache" not in path.parts)
        lines += [f"### {name}", ""]
        lines += [f"- `{path.relative_to(out)}`" for path in produced]
        lines.append("")
    lines += ["### Summary", ""]
    lines += [f"- `{name}`" for name in
              ("KEY_FINDINGS.md", "report.html", "report.body.html",
               "README.md", "run_metadata.json") if (out / name).exists()]
    lines.append("")
    path = out / "README.md"
    path.write_text("\n".join(lines))
    return path


# ---------------------------------------------------------------------------
# the HTML report
# ---------------------------------------------------------------------------
#: A topic below this spread across the dilemmas is one story's furniture
#: rather than a way of reading a person. 0 means it appears in one dilemma.
STORY_BOUND = 0.5


@dataclass
class ReportContext:
    """What the report needs about the run, read back from the output tree.

    The report is built from the CSVs rather than from the in-memory results,
    so `--sections report` can rebuild the page in a second without repeating
    thirty minutes of shuffling -- and so the page can never disagree with the
    tables it links to.
    """

    models: list
    n_topics: int
    outlier_share: float
    meta: dict
    story_bound: float = 0.0        # share of topics confined to few dilemmas
    median_spread: float = 0.0


def _read(out: Path, relative: str) -> pd.DataFrame:
    path = out / relative
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    return frame


def load_report_inputs(out: Path) -> tuple:
    """(context, one, two, three) from a completed output directory."""
    metadata = out / "run_metadata.json"
    if not metadata.exists():
        raise SystemExit(f"{metadata} is missing -- run the sections first")
    meta = json.loads(metadata.read_text())
    topics = _read(out, "shared/topics.csv")
    profile = _read(out, "shared/corpus_profile.csv")
    real = topics[topics["topic"] >= 0] if len(topics) else topics
    spread = (real["dilemma_spread"] if "dilemma_spread" in real.columns
              else pd.Series(dtype="float64"))
    context = ReportContext(
        models=list(profile["model"]) if len(profile) else [],
        n_topics=len(real),
        outlier_share=float(profile["outlier_share"].mean()) if len(profile) else 0.0,
        meta=meta,
        story_bound=float((spread < STORY_BOUND).mean()) if len(spread) else 0.0,
        median_spread=float(spread.median()) if len(spread) else 0.0,
    )
    zero = {
        "frequencies": _read(out, "00_language/term_frequencies.csv"),
        "contrasts": _read(out, "00_language/term_contrasts.csv"),
        "within": _read(out, "00_language/term_contrasts_within_side.csv"),
    }
    one = {
        "omnibus": _read(out, "01_persona_framing/omnibus_tests.csv"),
        "facets": _read(out, "01_persona_framing/persona_type_topic_contrasts.csv"),
        "framing": _read(out, "01_persona_framing/framing_topic_contrasts.csv"),
        "interaction": _read(out, "01_persona_framing/persona_by_framing_interaction.csv"),
        "metrics": _read(out, "01_persona_framing/response_metrics_by_condition.csv"),
        "references": _read(out, "01_persona_framing/reference_distances.csv"),
    }
    two = {
        "persona_words": _read(out, "02_persona_text/persona_words_by_topic.csv"),
        "persona_labels": _read(out, "02_persona_text/persona_words_by_assumption.csv"),
    }
    three = {
        "sensitivity": _read(out, "03_sycophancy/detectable_difference.csv"),
        "coverage": _read(out, "03_sycophancy/sycophancy_coverage.csv"),
        "terciles": _read(out, "03_sycophancy/topics_most_vs_least_sycophantic.csv"),
        "ranking": _read(out, "03_sycophancy/assumptions_ranked_by_sycophancy.csv"),
    }
    # Which models were left out is a fact the coverage table already records,
    # so the rebuilt prose reads it back rather than needing the run's memory.
    coverage = three["coverage"]
    three["skipped"] = (
        coverage.loc[coverage["sycophancy_model"].isna(), "assumptions_model"]
        .tolist() if "sycophancy_model" in coverage.columns else [])
    return context, zero, one, two, three

def _figure(out: Path, relative: str, caption: str):
    """A figure and its dark twin, or None when that figure was not produced."""
    light = out / relative
    dark = light.with_suffix(f".dark{light.suffix}")
    if not (light.exists() and dark.exists()):
        return None
    return rp.Figure(light=light, dark=dark, caption=caption)


def _figures(out: Path, items) -> str:
    made = [_figure(out, relative, caption) for relative, caption in items]
    return "".join(rp.figure_html(f) for f in made if f is not None)


def _range(values, fmt: str = "{:.3f}") -> str:
    """A per-model spread as one cell: "0.11-0.12", or the value if they agree."""
    values = list(values)
    if not values:
        return "-"
    low, high = min(values), max(values)
    if low == high:
        return fmt.format(low)
    return f"{fmt.format(low)}\u2013{fmt.format(high)}"


def _verdict(flags) -> str:
    """A card's state from what actually survived, across models.

    `flags` is one boolean per model. All of them is a result, none of them is
    a null, and a split is neither -- reporting a split as "detected" would let
    one model's finding stand in for both.
    """
    flags = list(flags)
    if not flags:
        return "descriptive"
    if all(flags):
        return "detected"
    if any(flags):
        return "mixed"
    return "none"


#: Terms kept per list. Enough to see a pattern, few enough to read; the CSVs
#: carry more.
EXPLORER_TOP = 20

#: How a dimension is named on the page. Anything not here is its column name
#: with the underscores taken out.
DIMENSION_LABELS = {
    "persona_type": "which part of their history was shown",
    "prompt_type": "which side of the story was told",
}


def _explorer_data(zero: dict, one: dict, two: dict, corpus) -> dict:
    """The tables the page reads, shaped for lookup in the browser.

    Two factors and nothing else. The persona type and the side of the story
    are what the design varies; the demographics file that ships with the data
    codes 200 people and the run draws 25, so a breakdown by any of its columns
    is two or three groups of eight and is not offered.

    Rounded and trimmed on the way out: the page reads shares to a tenth of a
    percent and z to one decimal, and shipping full float64 for twenty thousand
    rows triples the file for digits nobody displays.
    """
    frequencies = zero.get("frequencies")
    if frequencies is None or not len(frequencies):
        return {}
    within = zero.get("within")

    freq: dict = {}
    for row in frequencies.itertuples():
        if row.scope == "overall":
            key = "(everything)"
        elif row.scope == "slice":
            key = str(row.group)
        else:
            continue
        bucket = (freq.setdefault(row.model, {}).setdefault(row.level, {})
                  .setdefault(key, []))
        if len(bucket) < EXPLORER_TOP:
            bucket.append([row.term, round(float(row.share_assumptions), 4)])

    contrast: dict = {}
    if within is not None and len(within):
        for row in within.itertuples():
            key = f"{row.group}{SLICE_SEP}{row.prompt_type}"
            entry = (contrast.setdefault(row.model, {})
                     .setdefault(row.level, {})
                     .setdefault(key, {"ref": str(row.reference), "rows": []}))
            if len(entry["rows"]) < EXPLORER_TOP:
                entry["rows"].append([row.term, round(float(row.z), 1),
                                      round(float(row.share_group), 4),
                                      round(float(row.share_reference), 4)])

    personas = an.order_facets(
        [g for g in frequencies.loc[frequencies["scope"] == "persona_type",
                                    "group"].unique() if g != NO_PERSONA])
    sides = [side for side in (ORIGINAL, FLIPPED)
             if side in set(frequencies.loc[frequencies["scope"] == "prompt_type",
                                            "group"])]
    payload = {"models": list(corpus.models),
               "personas": [NO_PERSONA] + personas, "sides": sides,
               "sep": SLICE_SEP, "freq": freq, "contrast": contrast,
               "control": NO_PERSONA, "top": EXPLORER_TOP}
    _topic_payload(payload, one, two, corpus)
    return payload


def _topic_payload(payload: dict, one: dict, two: dict, corpus) -> None:
    """Fold the fitted topic space in as a third kind of term.

    The word and bigram panels count strings; this one counts the clusters
    those strings fall into, so the same two questions are answerable at both
    grains without a second set of controls.

    Topics are tabulated per persona type and per side, not per crossing of the
    two -- the topic contrasts come from section 1, which tests the two factors
    separately. A cell of the crossing therefore falls back to its persona
    type's row, and the page says which it is showing.
    """
    freq = payload["freq"]
    contrast = payload["contrast"]
    for table, column in ((one.get("facets"), "persona_type"),
                          (one.get("framing"), "prompt_type")):
        if table is None or not len(table):
            continue
        for (model, group), rows in table.groupby(["model", column]):
            rows = rows.copy()
            rows["topic"] = rows["topic"].astype(str)
            key = f"{column}:{group}"
            bucket = (freq.setdefault(str(model), {})
                      .setdefault("topic", {}).setdefault(key, []))
            for row in rows.nlargest(EXPLORER_TOP, "rate").itertuples():
                bucket.append([_short(row.topic, chars=42),
                               round(float(row.rate), 4)])
            ranked = rows.reindex(rows["z"].abs().sort_values(
                ascending=False).index)
            entry = (contrast.setdefault(str(model), {})
                     .setdefault("topic", {})
                     .setdefault(key, {"ref": "every other group", "rows": []}))
            for row in ranked.head(EXPLORER_TOP).itertuples():
                z = float(row.z) if np.isfinite(row.z) else 0.0
                survived = bool(pd.notna(row.q_normal) and row.q_normal < 0.05)
                entry["rows"].append([
                    _short(row.topic, chars=42) + (" *" if survived else ""),
                    round(z, 1), round(float(row.rate), 4),
                    round(float(row.rate_elsewhere), 4)])
    payload["hasTopics"] = any(
        "topic" in levels for levels in freq.values())


def _persona_chart(one: dict, corpus) -> str:
    """How far each disclosed slice of a history moves the model's read.

    Plotted as a deviation from each model's own mean, because the absolute
    distances differ between models for reasons that have nothing to do with
    the facets -- a shared axis in raw cosine would compare serving setups, not
    slices. The deviation is the comparable quantity and is what the section
    claims.
    """
    metrics = one.get("metrics")
    if metrics is None or not len(metrics):
        return ""
    block = metrics[metrics["metric"] == "dist_to_control"]
    if block.empty:
        return ""
    pivot = block.pivot_table(index="persona_type", columns="model",
                              values="mean")
    models = [m for m in corpus.models if m in pivot.columns]
    if not models:
        return ""
    relative = pivot[models] / pivot[models].mean() - 1.0
    order = [f for f in an.order_facets(relative.index) if f in relative.index]
    relative = relative.reindex(order)
    span = float(np.nanmax(np.abs(relative.to_numpy()))) or 1.0

    legend = "".join(
        f'<span class="key" style="background:var(--series-{i + 1})"></span>'
        f'<span>{rp.esc(model)}</span>' for i, model in enumerate(models))
    rows = []
    for facet, values in relative.iterrows():
        bars = []
        for index, model in enumerate(models):
            value = float(values[model])
            if not np.isfinite(value):
                continue
            width = abs(value) / span * 50.0
            side = ("left:50%" if value >= 0 else f"right:50%")
            bars.append(
                f'<span class="mini-bar" style="{side};width:{width:.1f}%;'
                f'background:var(--series-{index + 1})" '
                f'title="{rp.esc(model)}: {value:+.1%} against its own mean">'
                "</span>")
        rows.append(
            f'<div class="mini-row"><span class="mini-label">{rp.esc(facet)}'
            f'</span><span class="mini-track">{"".join(bars)}</span></div>')
    return f"""
  <figure class="chart">
    <p class="legend">{legend}</p>
    <div class="mini">{''.join(rows)}</div>
    <figcaption>How far each disclosed slice of a person's history moves the
    model's read of them, as a deviation from that model's own average slice.
    Bars right of the line move it further than average, left of it less.
    Plotted relative because the absolute distances differ between models for
    reasons unrelated to the slices. The raw numbers are in
    <code>01_persona_framing/response_metrics_by_condition.csv</code>.
    </figcaption>
  </figure>"""


def _persona_hits(zero: dict, corpus) -> str:
    """Which words each slice of a history puts into the model's reading.

    Word contrasts rather than topic contrasts, though both exist. A topic here
    is mostly one dilemma's furniture, so "reaches for chase / hannah / ai when
    hobbies is shown" is a fact about one story wearing the costume of a
    finding. The words are the thing the paper reports and they say something
    legible: disclose someone's politics and the model starts writing "civic",
    "participation", "engagement".

    Laid out model by model so replication is visible without a test for it:
    a row where all three columns say the same thing is worth more than the
    z-score of any one of them.
    """
    contrasts = zero.get("contrasts")
    if contrasts is None or not len(contrasts):
        return ""
    table = contrasts[(contrasts["dimension"] == "persona_type")
                      & (contrasts["level"] == "unigram")
                      & (contrasts["z"] > 0)]
    if table.empty:
        return ""
    models = [m for m in corpus.models if m in set(table["model"])]
    facets = [f for f in an.order_facets(table["group"].unique())
              if f in set(table["group"])]

    head = "".join(f"<th>{rp.esc(m)}</th>" for m in models)
    rows = []
    for facet in facets:
        cells = []
        for model in models:
            block = table[(table["model"] == model) & (table["group"] == facet)]
            top = block.nlargest(3, "z")
            cells.append("<td>" + (", ".join(
                f'<span class="term">{rp.esc(r.term)}</span>'
                f'<span class="z">+{r.z:.1f}</span>'
                for r in top.itertuples()) or "-") + "</td>")
        rows.append(f"<tr><th scope=\"row\">{rp.esc(facet)}</th>"
                    + "".join(cells) + "</tr>")
    return f"""
  <figure class="tablewrap words">
    <div class="scroll"><table>
      <thead><tr><th>slice shown</th>{head}</tr></thead>
      <tbody>{''.join(rows)}</tbody></table></div>
    <figcaption>The words each slice puts into the model's reading of the
    person, against the same dilemmas told with no history at all. Numbers are
    z-scored log-odds; past about 2 the difference is larger than sampling
    noise. Disclosing someone's politics makes two of the three models start
    writing "civic"; disclosing a hobby makes them write about the hobby. The
    full lists, and the bigram versions, are in
    <code>00_language/term_contrasts.csv</code>.</figcaption>
  </figure>"""


EXPLORER_SCRIPT = """
const D = window.__TERMS__;
const $ = (id) => document.getElementById(id);
const pct = (v) => (v * 100).toFixed(1) + '%';
const CONTROL = D.control;
const nice = (p) => (p === CONTROL ? 'no persona (control)' : p);

function fillSelect(node, values, labels) {
  node.innerHTML = '';
  values.forEach((value, index) => {
    const option = document.createElement('option');
    option.value = value;
    option.textContent = labels ? labels[index] : value;
    node.appendChild(option);
  });
}

function bars(host, rows, opts) {
  host.innerHTML = '';
  if (!rows || !rows.length) {
    host.innerHTML = '<p class="empty">' + (opts.empty || 'Nothing here.')
      + '</p>';
    return;
  }
  const span = Math.max(...rows.map((r) => Math.abs(opts.value(r)))) || 1;
  rows.forEach((row) => {
    const value = opts.value(row);
    const line = document.createElement('div');
    line.className = 'bar-row' + (opts.diverging ? ' diverging' : '');
    const term = document.createElement('span');
    term.className = 'bar-term';
    term.textContent = row[0];
    const track = document.createElement('span');
    track.className = 'bar-track';
    const fill = document.createElement('span');
    fill.className = 'bar-fill' + (opts.diverging && value < 0 ? ' negative' : '');
    fill.style.width = (Math.abs(value) / span * 100) + '%';
    track.appendChild(fill);
    const label = document.createElement('span');
    label.className = 'bar-value';
    label.textContent = opts.label(row);
    line.append(term, track, label);
    if (opts.title) line.title = opts.title(row);
    host.appendChild(line);
  });
}

function render() {
  const model = $('model').value;
  const level = $('level').value;
  const persona = $('persona').value;
  const side = $('side').value;
  const isTopic = level === 'topic';
  const byModel = D.freq[model] || {};
  const byLevel = byModel[level] || {};

  // Topics are tabulated per factor, not per crossing, so a topic view reads
  // the persona type's row and says so.
  const freqKey = isTopic ? 'persona_type:' + persona : persona + D.sep + side;
  const common = byLevel[freqKey] || [];
  bars($('common'), common, {
    value: (r) => r[1],
    label: (r) => pct(r[1]),
    empty: 'No terms for this combination.',
    title: (r) => r[0] + ' is ' + pct(r[1]) + ' of what was written here',
  });
  $('common-note').textContent = isTopic
    ? 'Share of the readings written when the persona type was "'
      + nice(persona) + '", both sides of the story pooled.'
    : 'Share of the assumptions written when the persona type was "'
      + nice(persona) + '" and the story was told as ' + side + '.';

  const entry = (((D.contrast[model] || {})[level] || {})[freqKey]) || null;
  const rows = entry ? entry.rows : [];
  const against = entry ? entry.ref : '';
  const isControl = persona === CONTROL;
  bars($('contrast'), rows, {
    diverging: true,
    value: (r) => r[1],
    label: (r) => (r[1] > 0 ? '+' : '') + r[1].toFixed(1),
    empty: 'No contrast for this combination.',
    title: (r) => r[0] + ': ' + pct(r[2]) + ' of what was written here, '
      + pct(r[3]) + ' of ' + against,
  });
  if (!rows.length) {
    $('contrast-note').textContent = '';
    $('legend').hidden = true;
    return;
  }
  $('contrast-note').textContent = isTopic
    ? 'Standard deviations from the permutation null, against every other '
      + 'persona type. A star marks the ones that survive correction across '
      + 'all topics tested.'
    : (isControl
        ? 'With no persona disclosed there is nothing to compare a persona '
          + 'against, so this is the other contrast the design offers: what '
          + 'this side of the story says more than ' + against + '.'
        : 'z-scored log-odds against the same dilemmas told the same way with '
          + 'no persona at all. Bars right are what disclosing "' + persona
          + '" adds; left is what it drops. Past about 2 the gap is bigger '
          + 'than sampling noise.');
  $('legend').hidden = false;
  $('legend-here').textContent = isTopic ? nice(persona)
    : (isControl ? side : nice(persona) + ', ' + side);
  $('legend-there').textContent = against;
}

fillSelect($('model'), D.models);
fillSelect($('persona'), D.personas, D.personas.map(nice));
fillSelect($('side'), D.sides);
if (!D.hasTopics) {
  const option = $('level').querySelector('option[value="topic"]');
  if (option) option.remove();
}
// Open on a disclosed persona rather than the control: both panels are
// populated and the contrast is the one the section is about.
if (D.personas.length > 1) $('persona').value = D.personas[1];
['model', 'level', 'persona', 'side'].forEach((id) =>
  $(id).addEventListener('change', render));
render();
"""


def explorer_html(zero: dict, one: dict, two: dict, corpus) -> tuple:
    """The interactive panel, and the JSON it reads."""
    data = _explorer_data(zero, one, two, corpus)
    if not data:
        return "", ""
    body = """
<section class="finding" id="words">
  <div class="section-head"><span class="section-number">01</span>
    <h2>What the assumptions say</h2></div>
  <div class="prose">
  <p>Every assumption is a sentence the model wrote about the person it was
  answering. This is what those sentences are made of: the most common words
  and bigrams, and &mdash; the part worth reading &mdash; which of them one
  setting uses more than another.</p>
  <p>The two panels answer different questions and each misleads alone. The
  left is what the model says most, and its top is boilerplate every condition
  shares. The right is what one condition says <em>more</em> than another, which
  is the comparison the study is about.</p>
  </div>
  <div class="controls">
    <label>Model <select id="model"></select></label>
    <label>Terms <select id="level">
      <option value="unigram">single words</option>
      <option value="2-gram">bigrams</option>
      <option value="topic">topics</option>
    </select></label>
    <label>Persona type <select id="persona"></select></label>
    <label>Side of the story <select id="side"></select></label>
  </div>
  <div class="panes">
    <div class="pane">
      <h3>Most common</h3>
      <p class="pane-note" id="common-note"></p>
      <div class="bars" id="common"></div>
    </div>
    <div class="pane">
      <h3>Used more here than there</h3>
      <p class="pane-note" id="contrast-note"></p>
      <p class="legend" id="legend" hidden>
        <span class="key key-here"></span><span id="legend-here"></span>
        <span class="key key-there"></span><span id="legend-there"></span>
      </p>
      <div class="bars" id="contrast"></div>
    </div>
  </div>
  <p class="prose small">Every number here is in
  <code>00_language/term_frequencies.csv</code> and
  <code>00_language/term_contrasts.csv</code>, which carry more terms per list
  than the page shows. Unigrams drop function words and bigrams keep them,
  following the paper's own tables.</p>
</section>"""
    script = ('<script>window.__TERMS__ = '
              + json.dumps(data, separators=(",", ":"))
              + ';</script>\n<script>' + EXPLORER_SCRIPT + '</script>')
    return body, script

def write_summaries(out: Path) -> list:
    """KEY_FINDINGS.md and report.html, both from the CSVs already written.

    File-backed so `--sections report` rebuilds them in a second against a
    finished run, and so neither can drift from the tables it cites.
    """
    context, zero, one, two, three = load_report_inputs(out)
    _write_language_notes(out / "00_language", zero)
    _write_section_notes(out / "01_persona_framing", one)
    _write_persona_text_notes(out / "02_persona_text", two)
    _write_sycophancy_notes(out / "03_sycophancy", three, context)
    findings = _key_findings(context, one, two, three, out, context.meta)
    report = write_report(out)
    return [findings, report, out / "report.body.html",
            out / "00_language" / "README.md",
            out / "01_persona_framing" / "README.md",
            out / "02_persona_text" / "README.md",
            out / "03_sycophancy" / "README.md",
            _index(out, context, context.meta)]


def write_report(out: Path) -> Path:
    """One page: the words first, the statistics after, no pictures of either.

    Reads the section CSVs back out of `out`, so it can be re-run on its own.

    Everything is drawn in HTML rather than embedded as an image. That keeps
    the page readable in both themes without shipping two renderings of every
    chart, keeps the numbers selectable, and lets the reader change what is
    plotted instead of scrolling past forty pictures of things they did not
    ask about. The figures on disk are still written for slides.
    """
    corpus, zero, one, two, three = load_report_inputs(out)
    meta = corpus.meta
    explorer, script = explorer_html(zero, one, two, corpus)

    models = " &middot; ".join(f"<b>{rp.esc(m)}</b>" for m in corpus.models)
    runline = " ".join([
        f"<span>models {models}</span>",
        f"<span>responses <b>{meta.get('n_responses', 0):,}</b></span>",
        f"<span>assumptions <b>{meta.get('n_assumptions', 0):,}</b></span>",
        f"<span>permutations <b>{meta.get('n_perm', 0):,}</b></span>",
        f"<span>{rp.esc(meta.get('generated', ''))}</span>",
    ])

    body = f"""
<header class="masthead"><div class="wrap">
  <p class="eyebrow">Verbalized assumptions &middot; open-ended probe</p>
  <h1>Who It Thinks You Are</h1>
  <p class="lede">Before answering someone's dilemma, each of these models was
  made to write down who it thought that person was &mdash; three ranked
  guesses, with a probability on each. The person's own chat history and their
  dilemma both came from a dataset; the guesses are the model's own words. This
  page is about what changed them.</p>
  <div class="runline">{runline}</div>
</div></header>
<main class="wrap">
  <nav class="index">
    <a href="#words">01 &nbsp;What the assumptions say</a>
    <a href="#slices">02 &nbsp;What the slice of history changes</a>
    <a href="#tests">03 &nbsp;What held up under test</a>
    <a href="#method">Method</a>
  </nav>
  {explorer}
  {_persona_section(zero, one, corpus)}
  {_statistics_section(corpus, one, two, three)}
  <section class="finding" id="method">
    <div class="section-head"><span class="section-number">&mdash;</span>
      <h2>How this was measured</h2></div>
    <div class="prose">
    <p>One cell of the design is one person, one slice of their chat history,
    one dilemma, and one of the two sides it can be told from. The model saw
    the history and the dilemma as text, wrote three candidate readings of the
    person with probabilities, then answered.</p>
    <p>The word and bigram tables are the paper's own descriptive pass, over
    the explanation the model gave for each reading. Unigrams drop function
    words and bigrams keep them, because the paper's bigram table reports
    "rather than" and "may have" and a stopword filter destroys those. The
    contrast is Monroe et al.'s z-scored log-odds with an informative prior
    taken from the whole corpus &mdash; the same estimator behind
    <code>syco text words</code>.</p>
    <p>The tests in section 03 use three different nulls, because the design gives each
    question a different unit of independence: which slice of history was shown
    and which side was told are shuffled inside each person-and-dilemma block;
    sycophancy is shuffled inside each dilemma.</p>
    </div>
    <div class="note">
      <p><strong>Reproduce it.</strong> One command writes every table, figure
      and word on this page.</p>
      <p><code>{rp.esc(meta.get('command', ''))}</code></p>
      <p>Rebuild just the prose and this page from the tables, without
      repeating the permutations:
      <code>python -m syco analyze --sections report --out {out}</code></p>
    </div>
    <ul class="filelist">
      <li><b>00_language/</b> word and bigram frequencies, and the contrasts
          behind the explorer above</li>
      <li><b>01_persona_framing/</b> omnibus tests, per-topic contrasts,
          McNemar, interaction, reference distances</li>
      <li><b>02_demographics/</b> coverage, omnibus, per-topic contrasts,
          top-5 lists, and the persona-transcript contrast</li>
      <li><b>03_sycophancy/</b> coverage, tercile contrasts, label ranking,
          per-topic regression</li>
      <li><b>KEY_FINDINGS.md</b> all of it in plain text, with the caveats</li>
    </ul>
  </section>
  <footer class="colophon">
    <p>Generated {rp.esc(meta.get('generated', ''))} by <code>python -m syco
    analyze</code>. Instrument after Cheng et al., <i>Verbalizing LLMs'
    assumptions</i>; sycophancy score after Neplenbroek et al.</p>
  </footer>
</main>
{script}"""

    title = "Who It Thinks You Are"
    path = out / "report.html"
    path.write_text(rp.standalone(title, body))
    (out / "report.body.html").write_text(rp.document(title, body))
    return path


def _persona_section(zero: dict, one: dict, corpus) -> str:
    """The facet finding, in two compact pieces and no prose about method."""
    chart = _persona_chart(one, corpus)
    hits = _persona_hits(zero, corpus)
    if not chart and not hits:
        return ""
    return f"""
<section class="finding" id="slices">
  <div class="section-head"><span class="section-number">02</span>
    <h2>What the slice of history changes</h2></div>
  <div class="prose">
  <p>Each person appears ten times over, once per slice of their chat history
  &mdash; their hobbies, their politics, a setback, what motivates them. The
  dilemma and the person are identical across the ten; only the slice differs,
  so anything that moves is the slice moving it.</p>
  </div>
  {chart}
  {hits}
</section>"""


def _persona_word_count(two: dict) -> str:
    """How many persona-word associations cleared the usual cut."""
    words = two.get("persona_words")
    if words is None or not len(words):
        return "-"
    return f"{int(_flags(words, 'above_threshold').sum())}/{len(words)}"


def _statistics_section(corpus, one: dict, two: dict, three: dict) -> str:
    """The three tests, as three sentences and three numbers each.

    Deliberately short. The tables on disk hold every cell; what belongs on a
    page is which questions got an answer and how big it was.
    """
    omnibus = one["omnibus"]
    facet = omnibus[(omnibus["factor"] == "persona_type")
                    & (omnibus["framing"] == "both")] if len(omnibus) else omnibus
    framing = omnibus[omnibus["factor"] == "prompt_type"] if len(omnibus) else omnibus
    references = one["references"]
    coverage = three["coverage"]
    scored = (coverage[coverage["sycophancy_model"].notna()]
              if len(coverage) and "sycophancy_model" in coverage else pd.DataFrame())
    skipped = (coverage.loc[coverage["sycophancy_model"].isna(), "assumptions_model"]
               .tolist() if len(coverage) and "sycophancy_model" in coverage else [])
    terciles = three["terciles"]
    syco_hits = (terciles[(terciles["tercile"] == "most sycophantic")
                          & (terciles["q_normal"] < 0.05)]
                 if len(terciles) else pd.DataFrame())

    distance = pd.DataFrame()
    if len(references):
        distance = references.pivot_table(index="model", columns="comparison",
                                          values="mean_distance")

    rows = []
    for model in corpus.models:
        if model not in distance.index:
            continue
        row = distance.loc[model]
        rows.append(
            f"<tr><td>{rp.esc(model)}</td>"
            f"<td class=\"num\">{row.get('same_person_other_facet', float('nan')):.3f}</td>"
            f"<td class=\"num\">{row.get('other_person_same_facet', float('nan')):.3f}</td>"
            f"<td class=\"num\">{row.get('cross_framing', float('nan')):.3f}</td></tr>")
    distance_table = ("" if not rows else f"""
  <figure class="tablewrap"><div class="scroll"><table>
    <thead><tr><th>model</th>
      <th class="num">same person, other slice</th>
      <th class="num">other person, same slice</th>
      <th class="num">same person, other telling</th></tr></thead>
    <tbody>{''.join(rows)}</tbody></table></div>
    <figcaption>Mean cosine distance between two readings that differ in
    exactly one thing. Which slice of a person's history is shown moves the
    model's read of them about as far as swapping in a different person;
    retelling the dilemma from the other side moves it further than
    either.</figcaption></figure>""")

    return f"""
<section class="finding" id="tests">
  <div class="section-head"><span class="section-number">03</span>
    <h2>What held up under test</h2></div>
  <div class="prose">
  <p>Three questions, three answers. Each is a randomization test against the
  design's own blocks, at {corpus.meta.get('n_perm', 0):,} shuffles; the tables
  on disk carry every cell and its corrected p-value.</p>
  </div>
  <div class="stats">
    {rp.stat(_range(facet['cramers_v'], '{:.2f}') if len(facet) else '-',
             'the slice of history shown',
             "Cramer's V on the mix of readings")}
    {rp.stat(_range(framing['cramers_v'], '{:.2f}') if len(framing) else '-',
             'the side the story is told from',
             'same measure, same corpus')}
    {rp.stat(_persona_word_count(two), 'persona words past z = 1.96',
             'in the transcripts, against what was assumed')}
    {rp.stat(f"{len(syco_hits)}" if len(scored) else 'n/a',
             'topics tied to sycophancy',
             (f"{len(set(scored['assumptions_model']))} of "
              f"{len(corpus.models)} models have a score of their own"))}
  </div>
  {distance_table}
  <div class="note">
    <p><strong>The demographics file is not used anywhere here.</strong> It
    codes each persona for age, gender, education, income and the rest, and
    this run draws 25 of its 200 people: every column lands as two or three
    groups of eight, which is not a comparison. What the persona side can
    support is the transcripts themselves, and that is section 02.</p>
  </div>
  <div class="note{' caveat' if skipped else ''}">
    <p><strong>Sycophancy is scored within-model only.</strong>{
      ' Skipped for want of a forced-choice collection of their own: '
      + ', '.join(f'<code>{rp.esc(m)}</code>' for m in skipped) + '.'
      if skipped else ''} Scoring a model against another model's collection
    would measure which dilemmas invite the behaviour while reading as if it
    were about the responding model, so it is not done.</p>
  </div>
</section>"""


# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="python -m syco analyze", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", action="append", default=None, metavar="ALIAS",
                   help="model alias; its table is results/<ALIAS>/"
                        f"{DEFAULT_PROBE}_assumptions.parquet. Repeatable. "
                        "Default: every alias that has a complete one")
    p.add_argument("--assumptions", action="append", default=None,
                   metavar="ALIAS=PATH",
                   help="an explicit table for one alias, instead of the "
                        "conventional path. Repeatable")
    p.add_argument("--probe", default=DEFAULT_PROBE,
                   help=f"probe label in the file names (default: {DEFAULT_PROBE})")
    p.add_argument("--personas", default="files/base_data_persona.gz",
                   help="persona transcripts, for the marked-word contrast "
                        "against what the model assumed")
    p.add_argument("--scores", action="append", default=None,
                   metavar="ALIAS=PATH",
                   help="one model's own forced-choice collection: a "
                        "*_results.pkl or a score table from `syco sycophancy "
                        "binary --out`. Repeatable. A bare PATH is matched to "
                        "a model by its file name. Default: look for each "
                        "analyzed model's own collection under "
                        "results/sycophancy/ and files/")
    p.add_argument("--score-dir", action="append", default=None,
                   help="extra directory to search for forced-choice "
                        "collections (default: results/sycophancy and files)")
    p.add_argument("--out", default=str(DEFAULT_OUT), help="output directory")
    p.add_argument("--sections", default="0,1,2,3,report",
                   help="which stages to run: 0, 1, 2, 3, and report (default: "
                        "all). `--sections report` rebuilds KEY_FINDINGS.md "
                        "and report.html from the CSVs already written, "
                        "without repeating the permutations")
    p.add_argument("--n-perm", type=int, default=2000,
                   help="permutation draws per test (default: 2000)")
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--min-label-count", type=int, default=10,
                   help="times an assumption label must be stated to be ranked")
    p.add_argument("--min-topic-size", type=int, default=50,
                   help="HDBSCAN minimum cluster size over the pooled corpus. "
                        "50 over ~60k assumptions leaves under a third "
                        "unclustered; raising it trades topics for outliers")
    p.add_argument("--nr-topics", default=None,
                   help="reduce to this many topics after fitting")
    p.add_argument("--stopwords", default="all",
                   choices=("all", "unigrams", "none"),
                   help="which n-gram levels drop function words. Default "
                        "`all`; `unigrams` is the paper's setting, which keeps "
                        "them in bigrams")
    p.add_argument("--keep-prompt-echo", action="store_true",
                   help="keep the words the probe puts in every answer "
                        f"({', '.join(sorted(PROMPT_ECHO))}). They are dropped "
                        "by default: the prompt names the person 'User A', so "
                        "counting them describes the prompt")
    p.add_argument("--field", default="description",
                   choices=("assumption", "description", "both"),
                   help="text clustered and embedded. Default `description`: "
                        "the explanation the model gave for each mental model, "
                        "which is also what the paper's own frequency tables "
                        "are computed over. `assumption` is the short label "
                        "alone -- fewer story-specific clusters, but a much "
                        "thinner string to cluster; `dilemma_spread` in "
                        "shared/topics.csv shows which topics are one story's "
                        "furniture either way")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    return p.parse_args(argv)


def _resolve_sources(args) -> dict:
    sources = {}
    for entry in args.assumptions or []:
        alias, _, path = entry.partition("=")
        if not path:
            raise SystemExit(f"--assumptions wants ALIAS=PATH, got {entry!r}")
        sources[alias] = path
    for alias in args.model or []:
        sources.setdefault(
            alias, str(RESULTS / alias / f"{args.probe}_assumptions.parquet"))
    if not sources:
        for directory in sorted(RESULTS.iterdir()):
            candidate = directory / f"{args.probe}_assumptions.parquet"
            if candidate.exists():
                sources[directory.name] = str(candidate)
    missing = [f"{a} -> {p}" for a, p in sources.items() if not Path(p).exists()]
    if missing:
        raise SystemExit(
            "no parsed assumptions table at:\n  " + "\n  ".join(missing)
            + f"\nRun: python -m syco parse results/<ALIAS>/{args.probe}.jsonl")
    if not sources:
        raise SystemExit(
            f"found no results/*/{args.probe}_assumptions.parquet. Parse a run "
            f"first: python -m syco parse results/<ALIAS>/{args.probe}.jsonl")
    return sources


def _warn_on_partial_runs(corpus: an.Corpus) -> None:
    """Say so when a model covers less of the design than its peers.

    With no `--model` the command takes every parsed table it finds, and a
    half-finished run looks exactly like a finished one once it is a parquet.
    The blocked tests drop its incomplete blocks safely, but its assumptions
    still shape the shared topic space and its rows still sit in every table,
    so this has to be visible rather than inferred from a row count later.
    """
    counts = corpus.responses.groupby("model").size()
    complete = counts.max()
    partial = counts[counts < complete]
    if partial.empty:
        return
    _say("\nwarning: these models cover less of the design than the fullest "
         f"one ({complete:,} responses):")
    for model, n in partial.items():
        _say(f"  {model}: {n:,} responses ({n / complete:.0%}). Finish the run "
             f"(`python -m syco run --model {model}`) or drop it from "
             "--model, unless a partial grid is what you meant.")
    _say("")


#: Where a model's own forced-choice collection is looked for.
SCORE_DIRECTORIES = (RESULTS / "sycophancy", Path("files"))

#: Filename shapes `syco sycophancy binary` and the collection itself use.
SCORE_PATTERNS = ("*_binary_sycophancy.parquet", "*_binary_sycophancy.csv",
                  "*_results.pkl")


def _normalize(name: str) -> str:
    return "".join(character for character in str(name).lower()
                   if character.isalnum())


def _score_stem(path: Path) -> str:
    """The model name a score file's own name claims."""
    stem = path.name
    for suffix in ("_binary_sycophancy.parquet", "_binary_sycophancy.csv",
                   "_long_results.pkl", "_results.pkl"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return path.stem


def discover_scores(models, explicit=None, extra_dirs=None) -> dict:
    """Match each analyzed model to *its own* forced-choice collection.

    Aliases and collection file names spell the same model differently
    (`Gemma3-12B` against `gemma-3-12b-it`), so both sides are reduced to
    lowercase alphanumerics and matched by containment. That is loose enough to
    pair the two and tight enough to keep `Llama-3.1-8B` away from a Gemma
    file; the resolved mapping is printed either way, because a wrong match
    here would silently produce a cross-model result that looks within-model.

    `*_long_results.pkl` is excluded by name: it holds free-text replies, not
    the constrained Yes/No log-probabilities the score is defined on.

    Returns {model_alias: (score_model_name, path)}. Models with no collection
    are simply absent.
    """
    found = {}
    for entry in explicit or []:
        alias, separator, path = entry.partition("=")
        if not separator:
            alias, path = None, entry
        candidate = Path(path)
        if not candidate.exists():
            raise SystemExit(f"--scores: no file at {candidate}")
        if alias:
            if alias not in models:
                raise SystemExit(
                    f"--scores {alias}=...: {alias} is not one of the analyzed "
                    f"models ({', '.join(models)})")
            found[alias] = (_score_stem(candidate), candidate)
        else:
            stem = _normalize(_score_stem(candidate))
            matched = [m for m in models if _normalize(m) in stem]
            if len(matched) != 1:
                raise SystemExit(
                    f"--scores {candidate}: matches {len(matched)} analyzed "
                    "models by name; pass it as ALIAS=PATH instead")
            found[matched[0]] = (_score_stem(candidate), candidate)

    directories = list(SCORE_DIRECTORIES) + [Path(d) for d in extra_dirs or []]
    for model in models:
        if model in found:
            continue
        target = _normalize(model)
        for directory in directories:
            if not directory.is_dir():
                continue
            for pattern in SCORE_PATTERNS:
                hits = [path for path in sorted(directory.glob(pattern))
                        if not path.name.endswith("_long_results.pkl")
                        and target in _normalize(_score_stem(path))]
                if hits:
                    found[model] = (_score_stem(hits[0]), hits[0])
                    break
            if model in found:
                break
    return found


def _load_scores(path) -> pd.DataFrame:
    scores = syc.load_scores(str(path))
    if scores is None:
        _say(f"  deriving binary sycophancy from {path} "
             "(this unpickles a large frame)")
        scores = syc.binary_scores(syc.load_binary(str(path)))
    return scores


def main(argv=None) -> int:
    args = parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    wanted = {s.strip() for s in args.sections.split(",") if s.strip()}
    if wanted == {"report"}:
        for path in write_summaries(out):
            _say(f"wrote {path}")
        return 0
    sources = _resolve_sources(args)

    _say(f"models: {', '.join(sources)}")
    corpus = an.build_corpus(
        sources, field=args.field, min_topic_size=args.min_topic_size,
        nr_topics=int(args.nr_topics) if args.nr_topics else None,
        seed=args.seed, device=args.device, threads=args.threads,
        cache=out / "shared" / "cache", embedding_model=args.embedding_model)
    _say(f"{corpus.meta['n_assumptions']:,} assumptions -> "
         f"{corpus.meta['n_responses']:,} responses, "
         f"{int((corpus.topics['topic'] >= 0).sum())} topics, "
         f"{corpus.meta['outlier_share']:.1%} unclustered")
    _warn_on_partial_runs(corpus)

    meta = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "command": "python -m syco " + " ".join(sys.argv[1:]),
        "sources": sources, "personas": args.personas, "scores": {},
        "n_perm": args.n_perm, "seed": args.seed,
        "n_responses": corpus.meta["n_responses"],
        "n_assumptions": corpus.meta["n_assumptions"],
        "topic_params": corpus.meta,
    }

    sections, results = {}, {}
    sections["Shared"] = _shared_tables(corpus, out)

    if "0" in wanted:
        _say("section 0: words and bigrams")
        results["zero"] = section_language(
            corpus, out / "00_language", stopword_mode=args.stopwords,
            extra_stopwords=frozenset() if args.keep_prompt_echo else PROMPT_ECHO)
        sections["0. Words and bigrams"] = results["zero"]["files"]

    if "1" in wanted:
        _say("section 1: persona facet and framing")
        results["one"] = section_persona_framing(
            corpus, out / "01_persona_framing", n_perm=args.n_perm, seed=args.seed)
        sections["1. Persona facet and framing"] = results["one"]["files"]

    if "2" in wanted:
        _say("section 2: the persona's own words")
        results["two"] = section_persona_text(
            corpus, out / "02_persona_text", persona_path=args.personas)
        sections["2. The persona's own words"] = results["two"]["files"]

    if "3" in wanted:
        _say("section 3: sycophancy")
        discovered = discover_scores(corpus.models, args.scores, args.score_dir)
        for model, (score_model, path) in sorted(discovered.items()):
            _say(f"  {model} <- {score_model} ({path})")
        meta["scores"] = {model: str(path)
                          for model, (_, path) in discovered.items()}
        loaded = {model: (score_model, _load_scores(path))
                  for model, (score_model, path) in discovered.items()}
        results["three"] = section_sycophancy(
            corpus, loaded, out / "03_sycophancy",
            n_perm=args.n_perm, seed=args.seed, min_count=args.min_label_count)
        sections["3. Sycophancy"] = results["three"]["files"]

    # Merged, not overwritten. A partial re-run (`--sections 1,report`) knows
    # nothing about the score mapping section 3 resolved last time, and
    # clobbering it would leave the rebuilt index claiming no collections were
    # found. Only the keys this invocation actually determined are replaced.
    metadata = out / "run_metadata.json"
    record = {}
    if metadata.exists():
        try:
            record = json.loads(metadata.read_text())
        except json.JSONDecodeError:
            record = {}
    fresh = {k: v for k, v in meta.items() if k != "topic_params"}
    if "3" not in wanted:
        fresh.pop("scores", None)
    # The recorded command has to be one that reproduces the whole directory.
    # A partial re-run's own argv does not, so it leaves the record alone.
    if not {"0", "1", "2", "3"} <= wanted and record.get("command"):
        fresh.pop("command", None)
    record.update(fresh)
    record["topic_params"] = {
        k: (v if isinstance(v, (int, float, str, type(None))) else str(v))
        for k, v in corpus.meta.items()}
    record["sections_last_run"] = sorted(wanted)
    metadata.write_text(json.dumps(record, indent=2))

    if "report" in wanted and (out / "run_metadata.json").exists():
        write_summaries(out)
    else:
        _index(out, corpus, meta)
    _say(f"\nwrote {out}/ -- start at {out}/KEY_FINDINGS.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
