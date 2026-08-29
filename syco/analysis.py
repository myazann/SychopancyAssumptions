"""Statistics for the open-ended assumptions grid.

Three questions are asked of the same table, and they need different tests
because the design gives each a different unit of independence:

1. **Persona facet and story framing.** Both are *within-subject*: the same
   person and the same dilemma appear under every facet and under both
   framings. The null is therefore not "the labels were drawn at random from
   the whole sample" but "inside this person-and-dilemma, the labels could have
   been attached to any of these responses". That is a randomization test with
   the design's own blocks, which is what `blocked_permutation` runs.

2. **Demographics.** A demographic attribute belongs to a *person*, not a
   response, and 25 people carry ~12,000 responses between them. A test that
   counts responses would report five-decimal p-values off an effective n of
   25. `cluster_permutation` shuffles the person-to-attribute map instead, so
   the null respects that.

3. **Sycophancy.** Sycophancy is a property of a (person, dilemma) cell and
   varies far more between dilemmas than between people, so every contrast here
   is taken *inside* a dilemma and pooled. `syco.sycophancy` already implements
   that for assumption labels; this module adds the tercile split and the
   cluster-robust logistic fit that read the same thing two other ways.

Everything in the module works on a **response**, not an assumption: a response
is one probe completion and carries k=3 verbalized assumptions, so a topic is
counted once per response that mentions it at all. Counting assumptions instead
would triple the sample size without adding an observation.

Nothing here interprets a topic. A topic is a cluster of wordings; that it
separates two groups is a fact about the model's language, not about the
groups.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from syco.data import NO_PERSONA
from syco.sycophancy import benjamini_hochberg
from syco.tables import normalize_label

__all__ = [
    "FACET_ORDER",
    "RESPONSE_KEYS",
    "Corpus",
    "assumption_text",
    "benjamini_hochberg",
    "blocked_permutation",
    "build_corpus",
    "chi_square",
    "cluster_permutation",
    "collapse_levels",
    "contrast_table",
    "cramers_v",
    "demographic_columns",
    "detectable_difference",
    "distance_to_control",
    "embed",
    "empirical_p",
    "fit_topics_cached",
    "jensen_shannon",
    "load_assumptions",
    "load_demographics",
    "load_persona_texts",
    "log_odds_ratio",
    "mcnemar",
    "normalized_entropy",
    "omnibus_from_permutation",
    "order_facets",
    "pairwise_reference_distances",
    "permutation_pvalues",
    "persona_words_by_topic",
    "response_table",
    "stratified_permutation",
    "term_contrasts",
    "topic_counts",
    "topic_indicators",
]

#: What identifies one probe completion once the model is a column of its own.
RESPONSE_KEYS = ("model", "persona_type", "persona_id",
                 "prompt_type", "prompt_id", "rep")

#: The persona facets in the order the source table lists them. Sorting puts
#: the `assumptions` facet first, which reads as if every persona had been
#: mislabeled as that one -- the same trap `syco.grid` documents.
FACET_ORDER = ("hobbies", "motivation", "recognition", "life_story", "crossroads",
               "family", "influence", "setback", "politics", "assumptions")

#: Laplace/Haldane correction for a log-odds ratio with an empty cell. Half a
#: count is the conventional choice and keeps the estimate finite without
#: moving a well-populated cell.
HALDANE = 0.5


# ---------------------------------------------------------------------------
# loading and shaping
# ---------------------------------------------------------------------------
def load_assumptions(sources: dict) -> pd.DataFrame:
    """Pool per-model `*_assumptions.parquet` tables into one long frame.

    `sources` maps a model alias to its parquet path. The alias becomes the
    `model` column; the tables are otherwise left exactly as `syco parse` wrote
    them, plus the shallow `label` normalization the rest of the repo shares.
    """
    frames = []
    for alias, path in sources.items():
        frame = pd.read_parquet(path)
        if "assumption" not in frame.columns:
            raise ValueError(f"{path} is not an open-ended assumptions table")
        frame = frame.copy()
        frame.insert(0, "model", str(alias))
        frame["label"] = frame["assumption"].map(normalize_label)
        frames.append(frame)
    if not frames:
        raise ValueError("no assumption tables given")
    pooled = pd.concat(frames, ignore_index=True)
    pooled["rep"] = pooled["rep"].astype(int)
    pooled["rank"] = pooled["rank"].astype(int)
    return pooled


def assumption_text(frame: pd.DataFrame, field: str = "both") -> pd.Series:
    """Label, description, or both as one string per assumption row."""
    label = frame["assumption"].fillna("").astype(str)
    if field == "assumption":
        return label
    desc = frame.get("description", pd.Series("", index=frame.index))
    desc = desc.fillna("").astype(str)
    if field == "description":
        return desc
    return label.str.cat(desc, sep=". ")


def embed(texts, *, model_name: str = "all-MiniLM-L6-v2", device: str = "auto",
          batch_size: int = 256, cache: Path | None = None,
          threads: int = 4) -> np.ndarray:
    """Sentence-transformer vectors, cached on the hash of the exact corpus.

    The cache key is the digest of the joined texts and the encoder name, so a
    changed corpus never silently reuses the previous run's vectors -- the two
    expensive steps downstream (the topic model and the distance-to-control
    metric) both read these, and a stale array would be invisible in both.
    """
    texts = [str(t) for t in texts]
    if cache is not None:
        digest = hashlib.sha256(
            ("\x00".join(texts) + "|" + model_name).encode()
        ).hexdigest()[:20]
        cache = Path(cache) / f"embeddings-{model_name.replace('/', '_')}-{digest}.npy"
        if cache.exists():
            return np.load(cache)
        cache.parent.mkdir(parents=True, exist_ok=True)

    import os

    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[variable] = str(threads)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch
    from sentence_transformers import SentenceTransformer

    from syco.topics import _resolve_device

    torch.set_num_threads(threads)
    encoder = SentenceTransformer(model_name, device=_resolve_device(device))
    vectors = encoder.encode(texts, batch_size=batch_size,
                             show_progress_bar=False,
                             convert_to_numpy=True).astype("float32")
    if cache is not None:
        np.save(cache, vectors)
    return vectors


def fit_topics_cached(frame: pd.DataFrame, embeddings: np.ndarray, *,
                      field: str = "assumption", min_topic_size: int = 50,
                      nr_topics=None, seed: int = 1000, device: str = "auto",
                      threads: int = 4, cache: Path | None = None):
    """`syco.topics.fit_topics` over the pooled corpus, with its two tables
    cached beside the embeddings.

    One topic space for every model and condition, deliberately: fitting per
    model would give each its own topics and leave nothing to compare across
    them. Outliers stay at topic -1 rather than being reassigned -- "the
    clusterer had no coherent topic for this" is a real answer, and folding it
    into the nearest topic is exactly the kind of thing that later reads as a
    finding.
    """
    from syco.topics import fit_topics

    key = None
    if cache is not None:
        digest = hashlib.sha256(
            f"{len(frame)}|{field}|{min_topic_size}|{nr_topics}|{seed}|"
            f"{embeddings.shape}|{float(embeddings[:64].sum()):.6f}".encode()
        ).hexdigest()[:20]
        cache = Path(cache)
        cache.mkdir(parents=True, exist_ok=True)
        key = cache / f"topics-{digest}"
        if (key.with_suffix(".info.parquet")).exists():
            info = pd.read_parquet(key.with_suffix(".info.parquet"))
            assigned = pd.read_parquet(key.with_suffix(".assign.parquet"))
            return (pd.Series(assigned["topic"].to_numpy(), index=frame.index,
                              name="topic"), info)

    result = fit_topics(frame, field=field, min_topic_size=min_topic_size,
                        nr_topics=nr_topics, seed=seed, device=device,
                        threads=threads, embeddings=embeddings)
    if key is not None:
        result.info.to_parquet(key.with_suffix(".info.parquet"), index=False)
        pd.DataFrame({"topic": result.assignments.to_numpy()}).to_parquet(
            key.with_suffix(".assign.parquet"), index=False)
    return result.assignments, result.info


def response_table(assumptions: pd.DataFrame,
                   embeddings: np.ndarray | None = None) -> pd.DataFrame:
    """One row per probe completion, with the per-response descriptives.

    * `top1_prob` -- the probability the model put on its leading mental model,
      renormalized. A response that puts 0.9 there has committed to one reading
      of the person; one that spreads 0.4/0.35/0.25 has not.
    * `prob_entropy` -- the same shape as a single number, normalized to [0, 1].
    * `n_topics` -- distinct topics among the response's k assumptions.
    * `topics` -- the set itself, as a tuple, for the indicator matrix.
    * `vector` index -- row of the response-level embedding matrix, the
      probability-weighted mean of its assumptions' vectors.
    """
    keys = [key for key in RESPONSE_KEYS if key in assumptions.columns]
    ordered = assumptions.sort_values(keys + ["rank"]).reset_index()
    grouped = ordered.groupby(keys, dropna=False, sort=True)

    rows = grouped.agg(
        n_assumptions=("rank", "size"),
        top1_prob=("probability_norm", "max"),
        parse_status=("parse_status", "first"),
    ).reset_index()

    probabilities = grouped["probability_norm"].apply(list)
    rows["prob_entropy"] = [normalized_entropy(p) for p in probabilities]

    if "topic" in ordered.columns:
        topics = grouped["topic"].apply(lambda s: tuple(sorted({int(t) for t in s})))
        rows["topics"] = topics.to_numpy()
        rows["n_topics"] = [len(t) for t in rows["topics"]]

    rows["text"] = grouped["_text"].apply(" ".join).to_numpy() \
        if "_text" in ordered.columns else ""

    if embeddings is not None:
        matrix = np.zeros((len(rows), embeddings.shape[1]), dtype="float32")
        # A handful of assumptions parse without a probability. Left as NaN the
        # weight poisons its whole response vector, and through the
        # distance-to-control lookup one bad control response then voids the
        # 250 responses that measure against it. A missing weight is treated as
        # no weight, and a response with no weights at all falls back to a
        # plain mean of its assumptions.
        weights = ordered["probability_norm"].to_numpy("float64")
        weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)
        source = ordered["index"].to_numpy()
        codes = grouped.ngroup().to_numpy()
        total = np.zeros(len(rows), dtype="float64")
        np.add.at(total, codes, weights)
        empty = total <= 0
        if empty.any():
            weights = np.where(empty[codes], 1.0, weights)
            total = np.zeros(len(rows), dtype="float64")
            np.add.at(total, codes, weights)
        np.add.at(matrix, codes, embeddings[source] * weights[:, None])
        matrix /= np.maximum(total, 1e-12)[:, None]
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        rows.attrs["vectors"] = matrix / np.maximum(norms, 1e-12)
    return rows


def topic_indicators(responses: pd.DataFrame, topics=None) -> tuple[np.ndarray, list]:
    """Response x topic 0/1 matrix: did this response mention this topic at all.

    Per response, not per assumption -- a response that reaches for the same
    topic twice is one response reaching for it.
    """
    if topics is None:
        topics = sorted({t for row in responses["topics"] for t in row})
    position = {topic: i for i, topic in enumerate(topics)}
    matrix = np.zeros((len(responses), len(topics)), dtype="float64")
    for row, assigned in enumerate(responses["topics"]):
        for topic in assigned:
            index = position.get(topic)
            if index is not None:
                matrix[row, index] = 1.0
    return matrix, list(topics)


@dataclass
class Corpus:
    """Everything the three analyses read, built once."""

    assumptions: pd.DataFrame
    responses: pd.DataFrame
    topics: pd.DataFrame
    indicators: np.ndarray
    topic_ids: list
    vectors: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    @property
    def models(self) -> list:
        return sorted(self.responses["model"].unique())

    def topic_name(self, topic: int) -> str:
        row = self.topics.loc[self.topics["topic"] == topic]
        if row.empty:
            return f"topic {topic}"
        return str(row.iloc[0].get("name", f"topic {topic}"))


def build_corpus(sources: dict, *, field: str = "assumption",
                 min_topic_size: int = 50,
                 nr_topics=None, seed: int = 1000, device: str = "auto",
                 threads: int = 4, cache: Path | None = None,
                 embedding_model: str = "all-MiniLM-L6-v2") -> Corpus:
    """Load, embed, cluster, and reduce to responses -- the shared front half.

    Run once and reused by all three analyses, because they must agree on the
    topic space: a topic that means one thing in the persona tables and another
    in the sycophancy tables makes the two uncomparable.
    """
    assumptions = load_assumptions(sources)
    texts = assumption_text(assumptions, field)
    assumptions["_text"] = texts

    vectors = embed(texts, model_name=embedding_model, device=device,
                    cache=cache, threads=threads)
    assignments, info = fit_topics_cached(
        assumptions, vectors, field=field, min_topic_size=min_topic_size,
        nr_topics=nr_topics, seed=seed, device=device, threads=threads,
        cache=cache)
    assumptions["topic"] = assignments.to_numpy()

    info = info.copy()
    info["name"] = [
        "(unclustered)" if int(t) < 0 else _short_name(words, examples)
        for t, words, examples in zip(info["topic"], info["top_words"],
                                      info.get("examples", [""] * len(info)))
    ]

    spread = (assumptions.groupby("topic")["prompt_id"]
              .apply(lambda s: normalized_entropy(s.value_counts(normalize=True)))
              .rename("dilemma_spread"))
    # 1.0 = the topic appears evenly across every dilemma, so it is about the
    # person; near 0 = it is one story's furniture ("the wedding", "the
    # barista"). The contrasts hold the dilemma fixed and are valid either way,
    # but a reader needs to know which kind of topic they are looking at.
    info = info.merge(spread.reset_index(), on="topic", how="left")

    responses = response_table(assumptions, vectors)
    indicators, topic_ids = topic_indicators(responses)
    return Corpus(
        assumptions=assumptions,
        responses=responses,
        topics=info,
        indicators=indicators,
        topic_ids=topic_ids,
        vectors=responses.attrs.get("vectors"),
        meta={"field": field, "min_topic_size": min_topic_size,
              "nr_topics": nr_topics, "seed": seed,
              "embedding_model": embedding_model,
              "n_assumptions": len(assumptions), "n_responses": len(responses),
              "outlier_share": float((assumptions["topic"] < 0).mean())},
    )


def topic_counts(corpus: Corpus) -> np.ndarray:
    """(n_responses, n_topics) -- how many of a response's assumptions landed
    in each topic.

    Every assumption has exactly one topic, so a row sums to k and the level
    totals downstream are proper multinomial counts. That is what makes the
    chi-square on a facet-by-topic table a real contingency test rather than a
    discrepancy measure over overlapping indicators.
    """
    keys = [key for key in RESPONSE_KEYS if key in corpus.assumptions.columns]
    codes = corpus.assumptions.groupby(keys, sort=True).ngroup().to_numpy()
    position = {topic: i for i, topic in enumerate(corpus.topic_ids)}
    columns = corpus.assumptions["topic"].map(position).to_numpy()
    matrix = np.zeros((len(corpus.responses), len(corpus.topic_ids)),
                      dtype="float64")
    np.add.at(matrix, (codes, columns), 1.0)
    return matrix


def distance_to_control(corpus: Corpus) -> pd.Series:
    """Cosine distance from each response to the no-persona response for the
    same model, dilemma, and framing.

    The direct reading of "how far did disclosing this identity move the
    model's picture of who it is talking to". It is a distance between two
    single draws at temperature 0.7, so a *level* of it means little; the
    comparison between facets, where the control is identical on both sides, is
    what the design supports.
    """
    if corpus.vectors is None:
        raise ValueError("corpus was built without response vectors")
    responses = corpus.responses
    control = responses["persona_type"] == NO_PERSONA
    keys = ["model", "prompt_id", "prompt_type"]
    reference = {}
    for position in np.flatnonzero(control.to_numpy()):
        row = responses.iloc[position]
        reference[tuple(row[key] for key in keys)] = corpus.vectors[position]

    out = np.full(len(responses), np.nan)
    coordinates = list(zip(*(responses[key] for key in keys)))
    for position, coordinate in enumerate(coordinates):
        anchor_vector = reference.get(coordinate)
        if anchor_vector is None or control.iat[position]:
            continue
        out[position] = 1.0 - float(corpus.vectors[position] @ anchor_vector)
    return pd.Series(out, index=responses.index, name="dist_to_control")


def pairwise_reference_distances(corpus: Corpus) -> pd.DataFrame:
    """Three reference distances per model, so `dist_to_control` has a scale.

    Within one dilemma and framing, how far apart are two responses when the
    only thing that changed was:

    * `same_person_other_facet` -- which facet of the same person was shown;
    * `other_person_same_facet` -- which person, with the facet held fixed;
    * `cross_framing` -- the telling, with person and facet held fixed.

    A distance is only interpretable against other distances. These three say
    whether moving the disclosed facet moves the model's picture as much as
    changing who the person is, which is the question behind analysis 1 and is
    unanswerable from `dist_to_control` alone.

    The mean pairwise cosine distance inside a group never needs the pairwise
    matrix. For unit vectors the off-diagonal similarities sum to
    ``||sum(v)||^2 - n``, so every group's mean comes from its vector sum --
    which is one grouped add. Building the 10,000 small matrices instead cost
    four minutes for these six numbers.
    """
    if corpus.vectors is None:
        raise ValueError("corpus was built without response vectors")
    responses = corpus.responses.reset_index(drop=True)
    vectors = np.asarray(corpus.vectors, dtype="float64")
    usable = (np.isfinite(vectors).all(axis=1)
              & (responses["persona_type"] != NO_PERSONA).to_numpy())
    frame = responses[usable].reset_index(drop=True)
    matrix = vectors[usable]
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.maximum(norms, 1e-12)

    def mean_within(group_keys, vary_key) -> pd.DataFrame:
        grouped = frame.groupby(group_keys, dropna=False, sort=False)
        codes = grouped.ngroup().to_numpy()
        n_groups = int(codes.max()) + 1 if len(codes) else 0
        if not n_groups:
            return pd.DataFrame(columns=["model", "distance", "pairs"])
        sums = np.zeros((n_groups, matrix.shape[1]), dtype="float64")
        np.add.at(sums, codes, matrix)
        size = np.bincount(codes, minlength=n_groups).astype("float64")
        distinct = grouped[vary_key].nunique().to_numpy()
        model = grouped["model"].first().to_numpy()

        keep = (size >= 2) & (distinct >= 2)
        pairs = size * (size - 1) / 2.0
        # sum of the off-diagonal similarities, from the group's vector sum
        off_diagonal = (sums ** 2).sum(axis=1) - size
        with np.errstate(invalid="ignore", divide="ignore"):
            similarity = off_diagonal / (size * (size - 1))
        return pd.DataFrame({"model": model[keep],
                             "distance": 1.0 - similarity[keep],
                             "pairs": pairs[keep]})

    frames = []
    for name, group_keys, vary in (
        ("same_person_other_facet",
         ["model", "persona_id", "prompt_id", "prompt_type"], "persona_type"),
        ("other_person_same_facet",
         ["model", "persona_type", "prompt_id", "prompt_type"], "persona_id"),
        ("cross_framing",
         ["model", "persona_type", "persona_id", "prompt_id"], "prompt_type"),
    ):
        table = mean_within(group_keys, vary)
        if table.empty:
            continue
        pooled = table.groupby("model").apply(
            lambda g: pd.Series({
                "mean_distance": float(np.average(g["distance"],
                                                  weights=g["pairs"])),
                "blocks": len(g),
            }), include_groups=False).reset_index()
        pooled["comparison"] = name
        frames.append(pooled)
    if not frames:
        return pd.DataFrame(columns=["model", "comparison", "mean_distance",
                                     "blocks"])
    out = pd.concat(frames, ignore_index=True)
    out["blocks"] = out["blocks"].astype(int)
    return out[["model", "comparison", "mean_distance", "blocks"]]


def _short_name(top_words, examples, n_terms: int = 4) -> str:
    """A readable handle for a topic, from its c-TF-IDF terms.

    Not a label in the paper's sense -- that step asks an LLM and is a separate
    command. This exists so a table of hundreds of topics is skimmable without
    joining it back to the topic frame every time.

    c-TF-IDF over unigrams *and* bigrams ranks "perfectionist", "perfectionist
    perfectionist" and "perfectionistic perfectionist" as three separate terms,
    and the naive join of the top four is a name that says one word four times.
    Repeated words are dropped inside each term and terms that add no new word
    are skipped, so four slots carry four ideas.
    """
    terms = [t.strip() for t in str(top_words or "").split(",") if t.strip()]
    seen: set = set()
    out = []
    for term in terms:
        words = list(dict.fromkeys(term.split()))
        fresh = [w for w in words if w not in seen]
        if not fresh:
            continue
        seen.update(words)
        out.append(" ".join(words))
        if len(out) >= n_terms:
            break
    if out:
        return " / ".join(out)
    first = str(examples or "").split(" | ")[0]
    return first[:40] or "(unnamed)"


# ---------------------------------------------------------------------------
# elementary statistics
# ---------------------------------------------------------------------------
def normalized_entropy(values) -> float:
    """Shannon entropy of a distribution, scaled so 1.0 is uniform."""
    p = np.asarray(list(values), dtype="float64")
    p = p[p > 0]
    if p.size < 2:
        return 0.0
    p = p / p.sum()
    return float(-(p * np.log(p)).sum() / np.log(p.size))


def jensen_shannon(p, q) -> float:
    """Jensen-Shannon divergence in bits, 0 identical to 1 disjoint."""
    p = np.asarray(p, dtype="float64")
    q = np.asarray(q, dtype="float64")
    if p.sum() <= 0 or q.sum() <= 0:
        return float("nan")
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float((a[mask] * np.log2(a[mask] / b[mask])).sum())

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def chi_square(table) -> tuple[float, int, float]:
    """Pearson chi-square of a contingency table -> (statistic, df, p)."""
    from scipy.stats import chi2_contingency

    table = np.asarray(table, dtype="float64")
    keep_rows = table.sum(axis=1) > 0
    keep_cols = table.sum(axis=0) > 0
    table = table[np.ix_(keep_rows, keep_cols)]
    if table.shape[0] < 2 or table.shape[1] < 2:
        return float("nan"), 0, float("nan")
    stat, p, dof, _ = chi2_contingency(table, correction=False)
    return float(stat), int(dof), float(p)


def cramers_v(table) -> float:
    """Chi-square rescaled to [0, 1] so tables of different size compare."""
    table = np.asarray(table, dtype="float64")
    stat, _, _ = chi_square(table)
    n = table.sum()
    k = min(table.shape) - 1
    if not np.isfinite(stat) or n <= 0 or k <= 0:
        return float("nan")
    return float(np.sqrt(stat / (n * k)))


def chi_square_statistic(table) -> float:
    """The bare statistic, for permutation loops.

    Expected counts come from the table's own margins on every draw, which is
    what makes the statistic comparable between the observed table and a
    permuted one.
    """
    table = np.asarray(table, dtype="float64")
    total = table.sum()
    if total <= 0:
        return 0.0
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / total
    mask = expected > 0
    return float(((table[mask] - expected[mask]) ** 2 / expected[mask]).sum())


def log_odds_ratio(a: float, b: float, c: float, d: float) -> tuple[float, float]:
    """log OR and its standard error for a 2x2 table, Haldane-corrected."""
    a, b, c, d = (float(a) + HALDANE, float(b) + HALDANE,
                  float(c) + HALDANE, float(d) + HALDANE)
    estimate = np.log((a * d) / (b * c))
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return float(estimate), float(se)


def mcnemar(b: int, c: int) -> float:
    """Exact two-sided McNemar p from the discordant pair counts.

    Exact rather than the chi-square approximation: a topic mentioned in 15
    responses out of 5,000 produces discordant counts in the single digits, and
    the approximation is unreliable exactly there.
    """
    from scipy.stats import binomtest

    n = int(b) + int(c)
    if n == 0:
        return 1.0
    return float(binomtest(int(b), n, 0.5, alternative="two-sided").pvalue)


def empirical_p(observed, null, two_sided: bool = True) -> np.ndarray:
    """Monte-Carlo p-value with the +1 correction.

    `null` is (n_perm, ...) and `observed` broadcasts against its tail axes.
    The +1 in both numerator and denominator keeps the p-value from ever being
    0, which for 2,000 draws it cannot honestly be.
    """
    observed = np.asarray(observed, dtype="float64")
    null = np.asarray(null, dtype="float64")
    n = null.shape[0]
    if two_sided:
        centre = null.mean(axis=0)
        extreme = (np.abs(null - centre) >= np.abs(observed - centre) - 1e-12)
    else:
        extreme = null >= observed - 1e-12
    return (extreme.sum(axis=0) + 1.0) / (n + 1.0)


# ---------------------------------------------------------------------------
# randomization tests that respect the design
# ---------------------------------------------------------------------------
def _balanced_tensor(values: np.ndarray, block: np.ndarray, level: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Rows -> a (block, level, feature) tensor. Blocks must be complete.

    Incomplete blocks are dropped rather than padded: a block missing a facet
    has no counterfactual for it, and filling one in with zeros would enter the
    permutation null as evidence that the facet is never used.
    """
    n_blocks = int(block.max()) + 1
    n_levels = int(level.max()) + 1
    n_features = values.shape[1]
    tensor = np.full((n_blocks, n_levels, n_features), np.nan, dtype="float64")
    tensor[block, level] = values
    complete = ~np.isnan(tensor).any(axis=(1, 2))
    return tensor[complete], np.flatnonzero(complete)


def blocked_permutation(values: np.ndarray, block: np.ndarray, level: np.ndarray,
                        *, group: np.ndarray | None = None, n_perm: int = 2000,
                        seed: int = 1000, batch: int = 8
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shuffle the level labels *inside* each block, `n_perm` times.

    This is the randomization test the design licenses. For the facet contrast
    a block is one (person, dilemma, framing) and the levels are the ten facets
    of that person, so the null is "which facet was disclosed carries no
    information", with the person and the dilemma held exactly fixed. For the
    framing contrast a block is one (facet, person, dilemma) and the two levels
    are the two tellings.

    `group` optionally splits the blocks -- pass the framing to get a separate
    table per framing out of one set of shuffles, which is what makes the
    facet-by-framing interaction testable: the same permutation has to be
    applied on both sides of the difference for the difference to have a null.

    Returns (observed, null, counts): observed is (n_groups, n_levels,
    n_features) of column *sums*, null is (n_perm, n_groups, n_levels,
    n_features), counts is (n_groups, n_levels) blocks behind each sum. Blocks
    are balanced by construction, so a sum divided by its count is a mean.
    """
    values = np.asarray(values, dtype="float64")
    if values.ndim == 1:
        values = values[:, None]
    tensor, kept = _balanced_tensor(values, np.asarray(block), np.asarray(level))
    n_blocks, n_levels, n_features = tensor.shape
    if n_blocks == 0:
        raise ValueError("no complete blocks: nothing to permute within")

    if group is None:
        block_group = np.zeros(n_blocks, dtype=int)
    else:
        group = np.asarray(group)
        by_block = np.zeros(int(np.asarray(block).max()) + 1, dtype=int)
        by_block[np.asarray(block)] = group
        block_group = by_block[kept]
    n_groups = int(block_group.max()) + 1
    masks = [block_group == g for g in range(n_groups)]

    observed = np.stack([tensor[mask].sum(axis=0) for mask in masks])
    counts = np.stack([np.full(n_levels, int(mask.sum())) for mask in masks])

    rng = np.random.default_rng(seed)
    null = np.empty((n_perm, n_groups, n_levels, n_features), dtype="float64")
    index = np.arange(n_blocks)[None, :, None]
    for start in range(0, n_perm, batch):
        size = min(batch, n_perm - start)
        # argsort of uniform noise is an independent permutation per block.
        order = np.argsort(rng.random((size, n_blocks, n_levels)), axis=2)
        shuffled = tensor[index, order]                # (size, blocks, lvl, feat)
        for g, mask in enumerate(masks):
            null[start:start + size, g] = shuffled[:, mask].sum(axis=1)
    return observed, null, counts


def cluster_permutation(values: np.ndarray, cluster: np.ndarray,
                        cluster_level: np.ndarray, *, n_perm: int = 2000,
                        seed: int = 1000
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shuffle an attribute across *clusters*, holding each cluster together.

    For the demographic tables a cluster is a person. Their attribute is fixed
    for all ~1,200 of their responses, so the responses are not exchangeable
    and only the 25 person-to-attribute assignments are. Permuting those is the
    honest null, and it is why these p-values are so much larger than a
    response-counting chi-square would give.

    Returns (observed, null, counts) shaped (n_levels, n_features),
    (n_perm, n_levels, n_features), and (n_levels,) responses per level.
    """
    values = np.asarray(values, dtype="float64")
    if values.ndim == 1:
        values = values[:, None]
    cluster = np.asarray(cluster)
    cluster_level = np.asarray(cluster_level)
    n_clusters = int(cluster.max()) + 1
    n_levels = int(cluster_level.max()) + 1

    sums = np.zeros((n_clusters, values.shape[1]), dtype="float64")
    sizes = np.zeros(n_clusters, dtype="float64")
    np.add.at(sums, cluster, values)
    np.add.at(sizes, cluster, 1.0)

    def table(levels):
        out = np.zeros((n_levels, values.shape[1]), dtype="float64")
        count = np.zeros(n_levels, dtype="float64")
        np.add.at(out, levels, sums)
        np.add.at(count, levels, sizes)
        return out, count

    observed, counts = table(cluster_level)
    rng = np.random.default_rng(seed)
    null = np.empty((n_perm, n_levels, values.shape[1]), dtype="float64")
    for draw in range(n_perm):
        null[draw] = table(rng.permutation(cluster_level))[0]
    return observed, null, counts


def stratified_permutation(values: np.ndarray, stratum: np.ndarray,
                           level: np.ndarray, *, n_perm: int = 2000,
                           seed: int = 1000
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shuffle level labels inside each stratum, with the group sizes unequal.

    `blocked_permutation` is the balanced case -- one response per level per
    block -- and is what the facet and framing contrasts have. The sycophancy
    terciles do not: a dilemma contributes however many of its cells fell in
    each third. This holds each stratum's group sizes fixed and reshuffles
    which responses filled them, so the dilemma's own difficulty cancels the
    same way while the arithmetic stays exact.

    Returns (observed, null, counts) shaped like `cluster_permutation`'s.
    """
    values = np.asarray(values, dtype="float64")
    if values.ndim == 1:
        values = values[:, None]
    stratum = np.asarray(stratum)
    level = np.asarray(level)
    n_levels = int(level.max()) + 1

    # Sort so each (stratum, level) is one contiguous run: a permuted table is
    # then a segment sum, which numpy does in one call, rather than a scatter
    # per row for every one of thousands of draws.
    order = np.lexsort((level, stratum))
    values = np.ascontiguousarray(values[order])
    stratum, level = stratum[order], level[order]

    starts = np.flatnonzero(np.concatenate(
        ([True], (stratum[1:] != stratum[:-1]) | (level[1:] != level[:-1]))))
    segment_level = level[starts]
    strata_starts = np.flatnonzero(np.concatenate(
        ([True], stratum[1:] != stratum[:-1])))
    strata_ends = np.append(strata_starts[1:], len(stratum))

    counts = np.zeros(n_levels, dtype="float64")
    np.add.at(counts, level, 1.0)

    def collapse(matrix):
        segments = np.add.reduceat(matrix, starts, axis=0)
        out = np.zeros((n_levels, matrix.shape[1]), dtype="float64")
        np.add.at(out, segment_level, segments)
        return out

    observed = collapse(values)
    rng = np.random.default_rng(seed)
    null = np.empty((n_perm, n_levels, values.shape[1]), dtype="float64")
    index = np.arange(len(values))
    for draw in range(n_perm):
        shuffled = index.copy()
        for start, end in zip(strata_starts, strata_ends):
            shuffled[start:end] = rng.permutation(shuffled[start:end])
        null[draw] = collapse(values[shuffled])
    return observed, null, counts


def omnibus_from_permutation(observed: np.ndarray, null: np.ndarray) -> dict:
    """Chi-square of a level x feature table against its permutation null.

    The asymptotic p-value is reported next to the permutation one on purpose:
    when they disagree by orders of magnitude, the gap *is* the dependence the
    design has and the asymptotic test ignores.
    """
    statistic = chi_square_statistic(observed)
    null_statistics = np.array([chi_square_statistic(t) for t in null])
    stat, dof, asymptotic = chi_square(observed)
    return {
        "chi2": statistic,
        "dof": dof,
        "p_asymptotic": asymptotic,
        "p_permutation": float((np.sum(null_statistics >= statistic - 1e-9) + 1)
                               / (len(null_statistics) + 1)),
        "cramers_v": cramers_v(observed),
        "n_perm": len(null_statistics),
        "_asymptotic_chi2": stat,
    }


def permutation_pvalues(observed: np.ndarray, null: np.ndarray, *,
                        min_expected: float = 5.0) -> dict:
    """Both p-values a set of shuffles supports, and which cells can use each.

    A Monte-Carlo p-value cannot fall below 1/(n_perm + 1). With a few thousand
    cells in the family that floor makes Benjamini-Hochberg unable to reject
    anything at all: at 2,000 draws the smallest attainable p is 5e-4, and over
    3,500 tests the smallest attainable q is above 1. Every cell then reports
    "not significant" for a reason that is about the number of shuffles drawn
    rather than about the data, which is the worst kind of null.

    The z-score against the same shuffles' own mean and standard deviation has
    no floor. For a count summed over hundreds of exchangeable blocks the
    permutation null is close to normal, so `p_normal` is what the correction
    runs on, and `p_mc` stays beside it as the assumption-free check.

    `usable` marks the cells where the approximation is safe -- the null has to
    vary at all, and the cell has to be common enough for the central limit
    theorem to have taken hold. Rare cells keep their MC p-value and are left
    out of the corrected family rather than given a q-value the normal
    approximation cannot support.
    """
    from scipy.stats import norm

    observed = np.asarray(observed, dtype="float64")
    null = np.asarray(null, dtype="float64")
    mean = null.mean(axis=0)
    sd = null.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(sd > 0, (observed - mean) / sd, np.nan)
    p_normal = 2.0 * norm.sf(np.abs(z))
    usable = (sd > 0) & (mean >= min_expected)
    return {"p_mc": empirical_p(observed, null), "null_mean": mean,
            "null_sd": sd, "z": z,
            "p_normal": np.where(usable, p_normal, np.nan), "usable": usable}


def contrast_table(observed: np.ndarray, null: np.ndarray, counts: np.ndarray,
                   level_names, feature_names, *, level_field: str = "level",
                   feature_field: str = "feature") -> pd.DataFrame:
    """Per (level, feature): rate, rate elsewhere, difference, and a p-value
    from the same shuffles the omnibus used.

    `lift` is the ratio and `delta` the difference; both are on the table
    because a topic at 1% that doubles and one at 40% that gains a point are
    different findings and each measure hides one of them.
    """
    observed = np.asarray(observed, dtype="float64")
    null = np.asarray(null, dtype="float64")
    counts = np.asarray(counts, dtype="float64")
    if counts.ndim == 1:
        counts = np.repeat(counts[:, None], observed.shape[1], axis=1)

    total = observed.sum(axis=0)
    total_n = counts.sum(axis=0)
    rest = total - observed
    rest_n = np.maximum(total_n - counts, 1e-12)

    rate = observed / np.maximum(counts, 1e-12)
    rate_rest = rest / rest_n
    tests = permutation_pvalues(observed, null)

    rows = []
    for i, level in enumerate(level_names):
        for j, feature in enumerate(feature_names):
            estimate, se = log_odds_ratio(
                observed[i, j], counts[i, j] - observed[i, j],
                rest[i, j], rest_n[i, j] - rest[i, j])
            rows.append({
                level_field: level,
                feature_field: feature,
                "n": float(counts[i, j]),
                "k": float(observed[i, j]),
                "rate": float(rate[i, j]),
                "rate_elsewhere": float(rate_rest[i, j]),
                "delta": float(rate[i, j] - rate_rest[i, j]),
                "lift": float(rate[i, j] / rate_rest[i, j])
                if rate_rest[i, j] > 0 else float("nan"),
                "log_or": estimate,
                "log_or_se": se,
                "expected_under_null": float(tests["null_mean"][i, j]),
                "null_sd": float(tests["null_sd"][i, j]),
                "z": float(tests["z"][i, j]),
                "p_normal": float(tests["p_normal"][i, j]),
                "p_monte_carlo": float(tests["p_mc"][i, j]),
                "testable": bool(tests["usable"][i, j]),
            })
    table = pd.DataFrame(rows)
    # Correct only over the cells the approximation can carry; the rest keep
    # their Monte-Carlo p and an explicit NaN rather than a q-value that would
    # be read as if it meant the same thing.
    q = np.full(len(table), np.nan)
    testable = table["testable"].to_numpy()
    if testable.any():
        q[testable] = benjamini_hochberg(table.loc[testable, "p_normal"])
    table["q_normal"] = q
    return table


def detectable_difference(contrast: pd.DataFrame, *, group_fields,
                          alpha: float = 0.05, power: float = 0.80
                          ) -> pd.DataFrame:
    """How large a difference each contrast could have found.

    A null result is only worth reading next to this. "No difference detected"
    over 25 people and "no difference detected" over 25,000 are the same
    sentence and completely different claims, and the number that separates
    them is the smallest effect the test had the power to see.

    The permutation null already carries it. Its standard deviation is the
    sampling noise of that cell's count under the design's own shuffling, so
    the smallest detectable count difference is (z_alpha/2 + z_power) * sd, and
    dividing by the level's denominator puts it in the same units as `delta`.

    Reported at the uncorrected `alpha`, which makes it the optimistic bound:
    the multiplicity correction the tables actually apply is stricter, so a
    real difference smaller than this was certainly not going to be found.
    """
    from scipy.stats import norm

    multiplier = norm.isf(alpha / 2) + norm.isf(1 - power)
    usable = contrast[contrast["testable"]].copy()
    if usable.empty:
        return pd.DataFrame()
    usable["min_detectable_delta"] = (
        multiplier * usable["null_sd"] / usable["n"].replace(0, np.nan))
    group_fields = list(group_fields)
    out = usable.groupby(group_fields, dropna=False).agg(
        topics_testable=("min_detectable_delta", "size"),
        median_rate=("rate", "median"),
        min_detectable_delta_median_topic=("min_detectable_delta", "median"),
        min_detectable_delta_best_topic=("min_detectable_delta", "min"),
        largest_observed_delta=("delta", lambda d: d.abs().max()),
    ).reset_index()
    out["alpha"] = alpha
    out["power"] = power
    return out


# ---------------------------------------------------------------------------
# demographics
# ---------------------------------------------------------------------------
#: The four vulnerability indices, deliberately screened out of the
#: demographic sweep. Each is a small integer count of vulnerability markers,
#: and the run draws 25 people: binning one gives two or three groups of eight,
#: and a rank correlation over 25 points is not an estimate worth reporting.
#: What the persona side of this study can actually support is
#: `persona_words_by_topic` -- the transcripts themselves, contrasted by what
#: the model went on to assume.
VULNERABILITY_COLUMNS = ("inherent_vulnerability", "situational_vulnerability",
                         "pathogenic_vulnerability", "vulnerability_score")

#: Columns of the demographics file that are identifiers, free text, or
#: screened out above.
DEMOGRAPHIC_SKIP = frozenset({"uuid", "occupation", "occupation_code",
                              *VULNERABILITY_COLUMNS})

#: Numeric-looking columns that are labels. Read as numbers they would be
#: correlated against, and a Spearman on an occupation code is nonsense.
CODE_COLUMNS = ("major_group_code",)


def load_demographics(path) -> pd.DataFrame:
    """The persona demographics/vulnerability file, keyed like the run tables."""
    demo = pd.read_csv(path)
    if "uuid" not in demo.columns:
        raise ValueError(f"{path} has no uuid column to join personas on")
    demo = demo.rename(columns={"uuid": "persona_id"})
    demo["persona_id"] = demo["persona_id"].astype(str)
    for column in CODE_COLUMNS:
        if column in demo.columns:
            demo[column] = demo[column].astype("string")
    return demo


def demographic_columns(demo: pd.DataFrame, persona_ids, *, min_per_level: int = 3,
                        min_levels: int = 2, max_levels: int = 8) -> pd.DataFrame:
    """Which demographic columns the sampled personas can actually support.

    The run draws 25 of the 200 people, so most columns arrive degenerate --
    one level, or six levels with two people in four of them. Reporting a test
    for those produces a table that looks like a result and is an artifact of
    the draw, so each column is screened here and the reason it was kept or
    dropped is written out next to the tests.
    """
    persona_ids = [p for p in dict.fromkeys(persona_ids) if p != NO_PERSONA]
    present = demo[demo["persona_id"].isin(persona_ids)]
    rows = []
    for column in demo.columns:
        if column == "persona_id" or column in DEMOGRAPHIC_SKIP:
            continue
        values = present[column]
        filled = values.dropna()
        counts = filled.value_counts()
        usable_levels = counts[counts >= min_per_level]
        numeric = pd.api.types.is_numeric_dtype(values)
        reason = ""
        if len(filled) < min_per_level * min_levels:
            reason = f"only {len(filled)} personas have a value"
        elif len(counts) < min_levels:
            reason = f"only {len(counts)} distinct value(s) among the sampled personas"
        elif len(usable_levels) < min_levels:
            reason = (f"no {min_levels} levels reach {min_per_level} personas "
                      f"(largest: {' / '.join(str(v) for v in counts.head(3))})")
        elif len(counts) > max_levels and not numeric:
            reason = f"{len(counts)} levels over {len(filled)} personas is too sparse"
        rows.append({
            "column": column,
            "kind": "numeric" if numeric else "categorical",
            "personas_with_value": len(filled),
            "levels": len(counts),
            "levels_with_min": len(usable_levels),
            "smallest_kept_level": int(usable_levels.min()) if len(usable_levels) else 0,
            "usable": not reason,
            "reason_dropped": reason,
            "level_counts": "; ".join(f"{k}={v}" for k, v in counts.items()),
        })
    return pd.DataFrame(rows).sort_values(
        ["usable", "levels_with_min"], ascending=[False, False]).reset_index(drop=True)


def load_persona_texts(path, role: str = "user") -> pd.DataFrame:
    """The persona transcripts, flattened to one string per (facet, person).

    `role="user"` keeps only the synthetic person's own turns. The assistant
    turns in these transcripts were written by whichever model built the
    persona, so counting their words as the person's would measure that model
    instead.
    """
    from syco.text_analysis import persona_texts

    frame = pd.read_pickle(path)
    frame = frame.dropna(subset=["persona_type", "persona_id"]).copy()
    frame["persona_type"] = frame["persona_type"].astype(str)
    frame["persona_id"] = frame["persona_id"].astype(str)
    frame["persona_text"] = persona_texts(frame, "persona_text", role=role)
    return frame[["persona_type", "persona_id", "persona_text"]]


def _count_matrix(texts) -> tuple[np.ndarray, list]:
    """(unit x vocabulary) counts, tokenized once.

    The sweep contrasts every topic against the same 250 transcripts, so the
    tokenization is done once into a dense matrix and each contrast is then a
    pair of row sums. Re-tokenizing per topic would make the pass quadratic for
    no gain.
    """
    from syco.text_analysis import token_counts

    counters = [token_counts([text]) for text in texts]
    vocabulary = sorted({word for counter in counters for word in counter})
    index = {word: i for i, word in enumerate(vocabulary)}
    matrix = np.zeros((len(counters), len(vocabulary)), dtype="float64")
    for row, counter in enumerate(counters):
        for word, count in counter.items():
            matrix[row, index[word]] = count
    return matrix, vocabulary


def _fightin_words(target: np.ndarray, reference: np.ndarray,
                   prior: np.ndarray) -> np.ndarray:
    """Monroe et al.'s z-scored log-odds with an informative Dirichlet prior.

    The same estimator as `syco.text_analysis.log_odds`, vectorized over the
    whole vocabulary at once so a sweep over hundreds of topics stays cheap.
    """
    a0 = np.maximum(prior, 1.0)
    a1, a2 = target + a0, reference + a0
    n1, n2 = a1.sum(), a2.sum()
    delta = np.log(a1 / (n1 - a1)) - np.log(a2 / (n2 - a2))
    return delta / np.sqrt(1.0 / a1 + 1.0 / a2)


def persona_words_by_topic(corpus: Corpus, personas: pd.DataFrame, *,
                           top: int = 10, min_units: int = 20,
                           threshold: float = 1.96) -> pd.DataFrame:
    """Which words in a person's own description go with what the model assumed.

    The unit is a (facet, person) transcript -- 250 of them, since the same
    person appears once per disclosed facet with different text. For each topic
    the units are split at the terciles of *how often the model gave that unit
    that topic*, and the top third's transcripts are contrasted against the
    bottom third's.

    A tercile split rather than "stated it at all": each unit has ~40 responses,
    so for any common topic almost every unit states it at least once and the
    presence/absence contrast has no comparison group left. Splitting on the
    rate keeps both sides populated and asks the sharper question anyway --
    what is different about the people the model reached for this reading of
    *most* often.

    This is an association in language, not a mechanism. The transcript and the
    assumption are both inputs to and outputs of the same conditioned model,
    and no ordering between a word and a topic is established here.
    """
    responses = corpus.responses.reset_index(drop=True)
    counts = topic_counts(corpus)
    keys = ["persona_type", "persona_id"]
    lookup = personas.drop_duplicates(subset=keys).set_index(keys)["persona_text"]

    rows = []
    for model in corpus.models:
        mask = ((responses["model"] == model)
                & (responses["persona_id"] != NO_PERSONA)).to_numpy()
        subset = responses[mask]
        unit = pd.MultiIndex.from_frame(subset[keys])
        codes, units = pd.factorize(unit)
        texts = [lookup.get(u, "") for u in units]
        usable = np.array([bool(str(t).strip()) for t in texts])
        if usable.sum() < 3 * min_units:
            continue
        matrix, vocabulary = _count_matrix(texts)
        vocabulary = np.asarray(vocabulary)

        per_unit = np.zeros((len(units), counts.shape[1]), dtype="float64")
        np.add.at(per_unit, codes, counts[mask])
        totals = np.bincount(codes, minlength=len(units)).astype("float64")
        rate = per_unit / np.maximum(totals, 1.0)[:, None]
        prior = matrix.sum(axis=0)

        for column, topic in enumerate(corpus.topic_ids):
            values = rate[:, column]
            eligible = usable & (totals > 0)
            if eligible.sum() < 3 * min_units:
                continue
            low, high = np.quantile(values[eligible], [1 / 3, 2 / 3])
            bottom = eligible & (values <= low)
            top_third = eligible & (values > high)
            if (bottom.sum() < min_units or top_third.sum() < min_units
                    or high <= low):
                continue
            z = _fightin_words(matrix[top_third].sum(axis=0),
                               matrix[bottom].sum(axis=0), prior)
            order = np.argsort(-np.abs(z))[:top]
            name = corpus.topic_name(int(topic))
            for rank, position in enumerate(order, 1):
                rows.append({
                    "model": model, "topic": f"T{int(topic)} {name}",
                    "rank": rank, "word": str(vocabulary[position]),
                    "z": float(z[position]),
                    "over_used_by": ("often assumed" if z[position] > 0
                                     else "rarely assumed"),
                    "above_threshold": bool(abs(z[position]) >= threshold),
                    "units_often": int(top_third.sum()),
                    "units_rarely": int(bottom.sum()),
                    "topic_rate_often": float(values[top_third].mean()),
                    "topic_rate_rarely": float(values[bottom].mean()),
                })
    return pd.DataFrame(rows)


def term_contrasts(assumptions: pd.DataFrame, dimension: str, *,
                   field: str = "description", reference: str | None = None,
                   levels=(1, 2), top: int = 30, min_count: int = 10,
                   stopword_mode: str = "all",
                   extra_stopwords: frozenset = frozenset()) -> pd.DataFrame:
    """Which words and bigrams one group uses *more* than another.

    The frequency tables answer "what does the model say"; almost every top
    term there is boilerplate the conditions share ("user a", "may be"). This
    answers the question the study is actually about -- what is said *more*
    here than there -- with Monroe et al.'s z-scored log-odds and an
    informative Dirichlet prior taken from the whole corpus, the same estimator
    `syco.text_analysis.marked_words` uses.

    Tokenization comes from `syco.topics.ngrams`, so a term here is the same
    string as in the frequency table and the two stack -- which means
    `stopword_mode` and `extra_stopwords` have to match whatever built that
    table, or the two tables are about different vocabularies.

    `reference` names the group everything is contrasted against -- the
    persona-free control for a facet contrast. With no reference each group is
    contrasted against every other group pooled.
    """
    from collections import Counter

    from syco.text_analysis import log_odds_from_counts
    from syco.topics import STOPWORD_MODES, ngrams

    if dimension not in assumptions.columns:
        raise ValueError(f"no {dimension!r} column to contrast on")
    if stopword_mode not in STOPWORD_MODES:
        raise ValueError(f"stopwords must be one of {sorted(STOPWORD_MODES)}")
    texts = assumption_text(assumptions, field)
    groups = assumptions[dimension].astype("string")
    drop_at = STOPWORD_MODES[stopword_mode]
    extra = frozenset(extra_stopwords)

    rows = []
    for n in levels:
        level = "unigram" if n == 1 else f"{n}-gram"
        counts: dict = {}
        for group, text in zip(groups, texts):
            if pd.isna(group):
                continue
            bucket = counts.setdefault(str(group), Counter())
            bucket.update(ngrams(text, n=n, drop_stopwords=n in drop_at,
                                 extra=extra))
        if len(counts) < 2:
            continue
        background: Counter = Counter()
        for bucket in counts.values():
            background.update(bucket)

        for group, bucket in counts.items():
            if reference is not None:
                if group == reference or reference not in counts:
                    continue
                other = counts[reference]
                against = reference
            else:
                other = background - bucket
                # With exactly two groups "everything else" *is* the other
                # group, and naming it beats printing a placeholder.
                against = (next(g for g in counts if g != group)
                           if len(counts) == 2 else "(every other group)")
            if not bucket or not other:
                continue
            z = log_odds_from_counts(bucket, other, background)
            total, other_total = sum(bucket.values()), sum(other.values())
            # Both tails: a term a group avoids is as much a finding as one it
            # reaches for, and a one-sided list hides half of every contrast.
            ranked = z.reindex(z.abs().sort_values(ascending=False).index)
            kept = 0
            for term, value in ranked.items():
                if bucket[term] + other[term] < min_count:
                    continue
                rows.append({
                    "dimension": dimension, "group": group,
                    "reference": against, "level": level, "term": term,
                    "z": float(value),
                    "used_more_by": group if value > 0 else against,
                    "n_group": int(bucket[term]),
                    "n_reference": int(other[term]),
                    "share_group": bucket[term] / total if total else 0.0,
                    "share_reference": other[term] / other_total
                    if other_total else 0.0,
                })
                kept += 1
                if kept >= top:
                    break
    return pd.DataFrame(rows)


def order_facets(values) -> list:
    """Present facets in the source table's order, unknown ones appended."""
    present = list(dict.fromkeys(values))
    known = [f for f in FACET_ORDER if f in present]
    return known + sorted(v for v in present if v not in FACET_ORDER)


def collapse_levels(values: pd.Series, min_per_level: int = 3) -> pd.Series:
    """Drop levels too small to compare; keep the rest verbatim.

    Deliberately not merged into an "other" bucket. "Other" would be a group
    the model was never told about, and a top-5 assumption list for it would
    describe nothing.
    """
    counts = values.value_counts()
    keep = set(counts[counts >= min_per_level].index)
    return values.where(values.isin(keep))
