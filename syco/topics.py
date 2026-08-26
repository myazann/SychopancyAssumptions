"""What the verbalized assumptions are *about*: n-grams and topic models.

`summarize_assumptions.py` counts assumption labels as exact strings, which
answers "how often does the model say exactly this" and nothing about content.
This module is the content pass, following Cheng et al., *Verbalizing LLMs'
assumptions to explain and control sycophancy* (vendored under
`verbalizedassumptions/`), which characterizes the open-ended assumptions three
ways:

1. **word frequency** -- "the word *validation* occurs in 26% of the
   assumptions for social sycophancy datasets on average";
2. **bigrams** -- "*Seeking validation* is the most frequent bigram [...]
   occurring in 12-16% of the responses";
3. **BERTopic** -- "we construct sentence-transformer embeddings for each
   assumption and use BERTopic to build topic models [...] then used GPT-4o to
   label each topic based on the top words".

Two of the paper's own choices are worth stating because they set the
denominators, and the paper reports each quantity against a different one:

* a **word** share is per *assumption* -- k rows per response, so a response
  with k=3 assumptions contributes three chances for the word to appear;
* a **bigram** share is per *response* -- one probe completion, counted once
  however many of its assumptions contain the bigram.

Both are reported for every term here rather than picking one, since the two
differ by roughly k and quoting the wrong one silently inflates a rate.

The paper fits one topic model per *dataset*. This study has one dilemma set
and varies the persona facet instead, so one topic model is fit over the whole
input and the facets are compared *within* that shared topic space. Fitting per
facet would give each facet its own topics and make the comparison meaningless
-- and comparing facets is the entire point of the design. That is also why
every table carries lift against the persona-free control: the paper's question
is what models assume, this design's question is what *disclosing a facet*
changes about what they assume.

Nothing here stems or lemmatizes, so "seeking"/"seeks" are distinct terms, and
nothing here merges near-synonymous labels -- that is what the topic model is
for.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import pandas as pd

from syco.data import NO_PERSONA
from syco.tables import cell_id, model_dimensions

#: The sentence-transformer the paper's method implies; small, and the default
#: for `sentence-transformers` itself.
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

#: Which text of an assumption to analyze. The probe emits a short label
#: (`model_name` in the paper's schema) and a sentence of description.
TEXT_FIELDS = ("assumption", "description", "both")

# Punctuation that ends a phrase: a bigram must not span it, or "...validation.
# They may..." yields the bigram "validation they". Hyphens are deliberately
# absent -- "people-pleasing" should yield the bigram "people pleasing".
_BREAK_RE = re.compile(r"""[.,;:!?()\[\]{}<>"“”`/\\|\n\r\t]+""")
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")

# Function words, for unigrams only. Bigrams keep them: the paper's own bigram
# table reports "rather than" and "may have", which a stopword filter destroys.
STOPWORDS = frozenset("""
a about above after again against all am an and any are aren't as at be because
been before being below between both but by can can't cannot could couldn't did
didn't do does doesn't doing don't down during each few for from further had
hadn't has hasn't have haven't having he he'd he'll he's her here here's hers
herself him himself his how how's i i'd i'll i'm i've if in into is isn't it
it's its itself let's me more most mustn't my myself no nor not of off on once
only or other ought our ours ourselves out over own same shan't she she'd she'll
she's should shouldn't so some such than that that's the their theirs them
themselves then there there's these they they'd they'll they're they've this
those through to too under until up very was wasn't we we'd we'll we're we've
were weren't what what's when when's where where's which while who who's whom
why why's with won't would wouldn't you you'd you'll you're you've your yours
yourself yourselves
""".split())  # noqa: SIM905 -- 170 words read as a block

#: `--stopwords` modes: which n-gram levels drop function words.
STOPWORD_MODES = {
    "unigrams": (1,),   # the paper's shape: content words, phrasal bigrams
    "all": (1, 2),
    "none": (),
}


# ---------------------------------------------------------------------------
# tokenizing
# ---------------------------------------------------------------------------
def segments(text) -> list[list[str]]:
    """Lowercased token runs, split at punctuation so bigrams stay phrasal."""
    if not isinstance(text, str) or not text.strip():
        return []
    out = []
    for chunk in _BREAK_RE.split(text.lower().replace("’", "'")):
        tokens = _TOKEN_RE.findall(chunk)
        if tokens:
            out.append(tokens)
    return out


def ngrams(text, n: int = 1, drop_stopwords: bool = False) -> list[str]:
    """The distinct n-grams of one piece of text, in order of first appearance.

    *Distinct*: every share this module reports is a share of assumptions or of
    responses that contain the term at all, never a token count, so a label
    that says "validation" twice must not count twice.
    """
    seen: dict[str, None] = {}
    for tokens in segments(text):
        if drop_stopwords:
            tokens = [t for t in tokens if t not in STOPWORDS]
        for i in range(len(tokens) - n + 1):
            seen.setdefault(" ".join(tokens[i:i + n]), None)
    return list(seen)


def assumption_text(df: pd.DataFrame, field: str = "assumption") -> pd.Series:
    """The text column to analyze. `both` keeps the label and its description
    as separate phrases, so no bigram straddles the join."""
    if field not in TEXT_FIELDS:
        raise ValueError(f"text field must be one of {TEXT_FIELDS}, got {field!r}")
    if field == "both":
        label = df["assumption"].fillna("").astype(str)
        desc = df.get("description", pd.Series("", index=df.index)).fillna("").astype(str)
        return label.str.cat(desc, sep=". ")
    if field not in df.columns:
        raise ValueError(f"table has no {field!r} column")
    return df[field].fillna("").astype(str)


# ---------------------------------------------------------------------------
# n-gram frequency
# ---------------------------------------------------------------------------
def _term_frame(df: pd.DataFrame, field: str, n: int,
                drop_stopwords: bool) -> pd.DataFrame:
    """Long frame: one row per (assumption row, distinct term)."""
    text = assumption_text(df, field)
    terms = text.map(lambda t: ngrams(t, n=n, drop_stopwords=drop_stopwords))
    base = pd.DataFrame({
        "_row": df.index,
        "_cell": cell_id(df).to_numpy(),
        "term": terms.to_numpy(),
    })
    return base.explode("term").dropna(subset=["term"])


def _shares(long: pd.DataFrame, group: list[str],
            totals: pd.DataFrame) -> pd.DataFrame:
    """Per group and term: how many assumptions and how many responses."""
    counts = (long.groupby([*group, "term"], dropna=False, observed=True)
              .agg(n_assumptions=("_row", "nunique"),
                   n_responses=("_cell", "nunique"))
              .reset_index())
    counts = counts.merge(totals, on=group, how="left") if group else counts.assign(
        **totals.iloc[0].to_dict())
    counts["share_assumptions"] = counts.n_assumptions / counts.assumptions
    counts["share_responses"] = counts.n_responses / counts.responses
    return counts


def ngram_frequencies(df: pd.DataFrame, *, field: str = "assumption",
                      levels=(1, 2), stopword_mode: str = "unigrams",
                      by=("persona_type",), top: int = 25,
                      min_count: int = 2) -> pd.DataFrame:
    """Term frequency per experiment, overall and within each grouping.

    One tidy table. `scope` is the grouping dimension a row belongs to --
    `overall` for the whole run, otherwise the column name -- and `group` is
    the level within it, so the paper's corpus-wide table and this study's
    per-facet contrast are the same schema and stack.

    `lift` compares a facet's share against the persona-free control's, and is
    NaN wherever there is no control to compare against.
    """
    if stopword_mode not in STOPWORD_MODES:
        raise ValueError(f"stopwords must be one of {sorted(STOPWORD_MODES)}")
    if df.empty:
        return pd.DataFrame()
    stopword_levels = STOPWORD_MODES[stopword_mode]
    dimensions = model_dimensions(df)
    scopes = [("overall", None), *((column, column) for column in by
                                   if column in df.columns)]

    frames = []
    for n in levels:
        long = _term_frame(df, field, n, drop_stopwords=n in stopword_levels)
        if long.empty:
            continue
        cells = pd.DataFrame({"_cell": cell_id(df).to_numpy()}, index=df.index)
        for scope, column in scopes:
            group = [*dimensions, *([column] if column else [])]
            keyed = long.join(df[group], on="_row") if group else long
            sized = cells.join(df[group]) if group else cells
            totals = (sized.groupby(group, dropna=False, observed=True)
                      .agg(assumptions=("_cell", "size"), responses=("_cell", "nunique"))
                      .reset_index()) if group else pd.DataFrame(
                [{"assumptions": len(cells), "responses": cells._cell.nunique()}])
            counts = _shares(keyed, group, totals)
            counts["level"] = "unigram" if n == 1 else f"{n}-gram"
            counts["scope"] = scope
            counts["group"] = counts[column] if column else "(all)"
            frames.append(counts.drop(columns=[column] if column else []))

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    # Lift first, then the rarity filter. A term the control used once is still
    # a denominator; dropping it first would null the lift on exactly the terms
    # a facet raised most, which is the column table 4 ranks on.
    out = _add_control_lift(out, dimensions)
    out = out[out.n_assumptions >= min_count]
    out = out.sort_values([*dimensions, "level", "scope", "group",
                           "share_assumptions"],
                          ascending=[True] * (len(dimensions) + 3) + [False])
    if top:
        out = out.groupby([*dimensions, "level", "scope", "group"],
                          dropna=False, observed=True, group_keys=False).head(top)
    columns = [*dimensions, "level", "scope", "group", "term",
               "n_assumptions", "assumptions", "share_assumptions",
               "n_responses", "responses", "share_responses",
               "control_share", "lift"]
    return out[[c for c in columns if c in out.columns]].reset_index(drop=True)


def _add_control_lift(table: pd.DataFrame, dimensions: list[str]) -> pd.DataFrame:
    """Attach each persona facet's term share relative to the control's."""
    table = table.copy()
    table["control_share"] = float("nan")
    table["lift"] = float("nan")
    facet = table[(table.scope == "persona_type") & (table.group == NO_PERSONA)]
    if facet.empty:
        return table
    keys = [*dimensions, "level", "term"]
    base = facet.set_index(keys).share_assumptions
    mask = table.scope == "persona_type"
    lookup = pd.MultiIndex.from_frame(table.loc[mask, keys])
    control = base.reindex(lookup).to_numpy()
    table.loc[mask, "control_share"] = control
    table.loc[mask, "lift"] = _ratio(table.loc[mask, "share_assumptions"].to_numpy(),
                                     control)
    return table


def _ratio(numerator, denominator):
    """share / control_share, NaN wherever the control offers no denominator."""
    import numpy as np

    denominator = np.asarray(denominator, dtype=float)
    safe = np.where((denominator > 0) & np.isfinite(denominator), denominator, np.nan)
    return np.asarray(numerator, dtype=float) / safe


def corpus_profile(df: pd.DataFrame, field: str = "assumption") -> pd.DataFrame:
    """Size of the corpus each frequency table is computed over.

    `mean_tokens` follows `field`; `distinct_labels` always counts the short
    assumption labels, since that is the quantity the exact-string frequency
    tables in `summarize_assumptions.py` key on and what the topic model exists
    to look past.
    """
    text = assumption_text(df, field)
    dimensions = model_dimensions(df)
    frame = pd.DataFrame({
        "_cell": cell_id(df).to_numpy(),
        "_tokens": text.map(lambda t: sum(len(s) for s in segments(t))).to_numpy(),
        "_label": df["assumption"].fillna("").astype(str).str.lower().to_numpy(),
    }, index=df.index)
    if dimensions:
        frame = frame.join(df[dimensions])
    grouped = frame.groupby(dimensions, dropna=False, observed=True) if dimensions \
        else [((), frame)]
    rows = []
    for value, sub in grouped:
        values = value if isinstance(value, tuple) else (value,)
        rows.append({
            **dict(zip(dimensions, values)),
            "assumptions": len(sub),
            "responses": sub._cell.nunique(),
            "distinct_labels": sub._label.nunique(),
            "mean_tokens": round(float(sub._tokens.mean()), 2),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# BERTopic
# ---------------------------------------------------------------------------
#: Below this many assumptions, UMAP + HDBSCAN has nothing to find and returns
#: one topic or all outliers. Better to say so than to print a table of noise.
MIN_DOCUMENTS = 30

INSTALL_HINT = ("pip install -r requirements.txt "
                "(the bertopic and sentence-transformers lines)")


@dataclass
class TopicResult:
    """A fitted topic model, reduced to the two tables the analysis needs."""
    assignments: pd.Series          # topic id per assumption row, indexed like df
    info: pd.DataFrame              # one row per topic: size, share, top words
    outlier_share: float            # share of assumptions BERTopic left at -1
    params: dict = field(default_factory=dict)

    @property
    def n_topics(self) -> int:
        return int((self.info.topic >= 0).sum())


def topics_available() -> tuple[bool, str]:
    """Whether the optional topic-modeling stack is importable."""
    import importlib.util

    missing = [name for name in
               ("bertopic", "sentence_transformers", "umap", "sklearn")
               if importlib.util.find_spec(name) is None]
    if missing:
        return False, f"missing {', '.join(missing)}; {INSTALL_HINT}"
    return True, ""


# Pool-size variables the numerical stack reads once, at its own import time.
_POOL_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                   "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS")


#: Free VRAM a card must have before `--device auto` will take it. A
#: MiniLM-class encoder needs a fraction of this; the headroom is because a
#: CUDA context alone costs a few hundred MiB, and the whole point of the check
#: is to leave a card a generation run is holding alone. Falling back to CPU
#: costs seconds, so the threshold errs high.
MIN_FREE_VRAM_MIB = 2048


def _visible_gpus() -> list[int]:
    """Physical GPU indices in the order torch will number them.

    `CUDA_VISIBLE_DEVICES` remaps them, so torch's `cuda:0` is not necessarily
    nvidia-smi's GPU 0. Free memory is only known per *physical* card, so the
    two have to be lined up before a device string can be built.
    """
    import os

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return []                                # unset: torch order == smi order
    return [int(part) for part in visible.split(",") if part.strip().isdigit()]


def _resolve_device(device: str) -> str:
    """`auto` -> the emptiest GPU with room, else CPU. Anything else is honored.

    The free-memory check comes from `nvidia-smi`, via the same helper the
    scheduler uses to place models, for two reasons. It is the number that
    actually matters -- "is a generation run holding this card" -- and unlike
    `torch.cuda.mem_get_info` it does not have to initialize a CUDA context on
    each card to ask, which would itself claim a few hundred MiB on a GPU the
    analysis has just decided not to use.
    """
    if device != "auto":
        return device
    try:
        from syco.orchestrate import query_gpus  # lazy: no import cycle

        gpus = query_gpus()
    except (RuntimeError, ImportError):
        return "cpu"                                 # no NVIDIA GPUs, or no smi

    free_by_index = {gpu.index: gpu.free_mib for gpu in gpus}
    order = _visible_gpus() or sorted(free_by_index)
    ranked = [(free_by_index.get(physical, 0), position)
              for position, physical in enumerate(order)]
    if not ranked:
        return "cpu"
    free_mib, position = max(ranked)
    return f"cuda:{position}" if free_mib >= MIN_FREE_VRAM_MIB else "cpu"


def _limit_thread_pools(threads: int) -> None:
    """Cap every BLAS/OpenMP pool before the libraries that read them import.

    Each of torch, OpenBLAS, MKL and numba sizes its pool to the core count.
    That is fine on a workstation and fatal on a shared machine: the systemd
    user slice here caps the whole login at 1024 pids, the generation runs hold
    most of them, and four pools of 48 on top is what makes this process die on
    pthread_create rather than on anything to do with the data.

    The imports these govern all happen inside `fit_topics`, so setting them
    here lands in time. numpy's own BLAS pool is the exception -- pandas pulls
    numpy in at module import, long before this -- which is why
    `scripts/topic_assumptions.py` sets a default for it above its imports.
    """
    import os

    for variable in _POOL_VARIABLES:
        os.environ[variable] = str(threads)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def fit_topics(df: pd.DataFrame, *, field: str = "assumption",
               embedding_model: str = DEFAULT_EMBEDDING_MODEL,
               min_topic_size: int = 10, nr_topics=None, seed: int = 1000,
               reduce_outliers: bool = False, top_words: int = 10,
               device: str = "auto", threads: int = 4) -> TopicResult:
    """Sentence-transformer embeddings -> BERTopic, over the whole input.

    `seed` is threaded into UMAP's `random_state`. Without it BERTopic is not
    reproducible, and a topic table that changes between two runs of the same
    command cannot be reported.

    `device` defaults to CPU and `threads` to 4, which is a deliberate choice
    about what this step is: an analysis pass that runs on the same machine as
    the generation runs, over a few thousand short strings. Left to their
    defaults, sentence-transformers takes whichever GPU it can see -- the one
    the scheduler is holding for a 12B model -- and torch, OpenBLAS and numba
    each size their pools to all cores, which is what exhausts a shared box.
    Embedding this corpus on four CPU threads takes seconds. Pass
    `device="cuda"` when nothing else is competing for the card.

    Outliers (topic -1) are kept by default and reported as `outlier_share`.
    BERTopic can reassign them, but doing that silently turns "the model had no
    coherent topic for this assumption" into a topic membership, which is
    exactly the kind of thing that later reads as a finding. `outlier_share` is
    always the rate *before* any reassignment, so `reduce_outliers=True` leaves
    no topic -1 in `info` while the share still says how much was moved.
    """
    ok, why = topics_available()
    if not ok:
        raise RuntimeError(f"topic modeling unavailable: {why}")
    _limit_thread_pools(threads)
    if len(df) < MIN_DOCUMENTS:
        raise RuntimeError(
            f"{len(df)} assumptions is too few to fit topics "
            f"(need >= {MIN_DOCUMENTS}); the n-gram tables still apply")

    try:
        import torch
        from bertopic import BERTopic
        from hdbscan import HDBSCAN
        from sentence_transformers import SentenceTransformer
        from sklearn.feature_extraction.text import CountVectorizer
        from umap import UMAP
    except ImportError as err:                                 # half-installed
        # find_spec found the packages but one of them cannot import -- an
        # interrupted or mismatched install, which is worth naming rather than
        # letting it surface as a bare ModuleNotFoundError three frames down.
        raise RuntimeError(
            f"the topic stack is installed but not importable ({err}). "
            f"Re-run: {INSTALL_HINT}"
        ) from err

    # The environment cap above governs torch's OpenMP pool; its intra-op pool
    # is a separate setting, and encoding is where it would be spent.
    torch.set_num_threads(threads)

    resolved = _resolve_device(device)
    docs = assumption_text(df, field).tolist()
    try:
        encoder = SentenceTransformer(embedding_model, device=resolved)
    except Exception as err:
        # sentence-transformers fetches weights on first use, so this is a
        # download or a bad model name -- both are "the topic tables cannot be
        # produced", not "the n-gram tables are wrong".
        raise RuntimeError(
            f"could not load sentence-transformer {embedding_model!r}: "
            f"{type(err).__name__}: {err}"
        ) from err
    embeddings = encoder.encode(docs, show_progress_bar=False)

    # UMAP's default n_neighbors=15 exceeds the corpus on small runs.
    neighbors = max(2, min(15, len(docs) - 1))
    umap = UMAP(n_neighbors=neighbors, n_components=5, min_dist=0.0,
                metric="cosine", random_state=seed)
    # c-TF-IDF top words, stopword-free, matching the paper's reported top
    # words ("seeking, feels, actions, emotional, validation").
    vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 2),
                                 min_df=1)
    # A min_topic_size larger than a tenth of the corpus leaves HDBSCAN able
    # to form at most a handful of clusters, so it is clamped -- and the value
    # actually used is reported, since a silently overridden setting is worse
    # than a rejected one.
    effective_min_size = min(min_topic_size, max(2, len(docs) // 10))
    # BERTopic's default HDBSCAN fans core-distance computation out over every
    # core; built explicitly so `threads` governs that too.
    clusterer = HDBSCAN(min_cluster_size=effective_min_size, metric="euclidean",
                        cluster_selection_method="eom", prediction_data=True,
                        core_dist_n_jobs=threads)
    model = BERTopic(embedding_model=encoder, umap_model=umap,
                     hdbscan_model=clusterer, vectorizer_model=vectorizer,
                     min_topic_size=effective_min_size,
                     nr_topics=nr_topics, calculate_probabilities=False,
                     verbose=False)
    assigned, _ = model.fit_transform(docs, embeddings)
    outlier_share = float(sum(t == -1 for t in assigned) / len(assigned))
    if reduce_outliers and -1 in set(assigned):
        assigned = model.reduce_outliers(docs, assigned, strategy="embeddings",
                                         embeddings=embeddings)
        model.update_topics(docs, topics=assigned, vectorizer_model=vectorizer)

    assignments = pd.Series(assigned, index=df.index, name="topic")
    info = _topic_info(model, assignments, df, top_words)
    return TopicResult(
        assignments=assignments,
        info=info,
        outlier_share=outlier_share,
        params={
            "embedding_model": embedding_model,
            "field": field,
            "min_topic_size": min_topic_size,
            "min_topic_size_effective": effective_min_size,
            "nr_topics": nr_topics,
            "seed": seed,
            "reduce_outliers": reduce_outliers,
            "device": resolved,
            "n_documents": len(docs),
        },
    )


def _topic_info(model, assignments: pd.Series, df: pd.DataFrame,
                top_words: int) -> pd.DataFrame:
    """One row per topic: size, both shares, top words, representative labels."""
    cells = cell_id(df)
    labels = df["assumption"].fillna("").astype(str)
    rows = []
    for topic in sorted(set(assignments)):
        mask = assignments == topic
        words = model.get_topic(topic) or []
        examples = labels[mask].value_counts().head(3).index.tolist()
        rows.append({
            "topic": int(topic),
            "n_assumptions": int(mask.sum()),
            "share_assumptions": float(mask.mean()),
            "n_responses": int(cells[mask].nunique()),
            "share_responses": float(cells[mask].nunique() / cells.nunique()),
            "top_words": ", ".join(w for w, _ in words[:top_words]),
            "examples": " | ".join(examples),
        })
    return pd.DataFrame(rows).sort_values("n_assumptions", ascending=False)


def topic_shares(df: pd.DataFrame, assignments: pd.Series,
                 by: str = "persona_type") -> pd.DataFrame:
    """Topic distribution within each level of `by`, with lift vs. the control.

    Shares are per assumption and sum to 1 within a group (outliers included as
    topic -1), so a facet's row reads as "this much of what this facet made the
    model assume fell in this topic".
    """
    if by not in df.columns:
        return pd.DataFrame()
    dimensions = model_dimensions(df)
    frame = df[[*dimensions, by]].copy()
    frame["topic"] = assignments
    frame["_cell"] = cell_id(df).to_numpy()

    group = [*dimensions, by]
    counts = (frame.groupby([*group, "topic"], dropna=False, observed=True)
              .agg(n_assumptions=("topic", "size"), n_responses=("_cell", "nunique"))
              .reset_index())
    totals = (frame.groupby(group, dropna=False, observed=True)
              .agg(assumptions=("topic", "size"), responses=("_cell", "nunique"))
              .reset_index())
    counts = counts.merge(totals, on=group, how="left")
    counts["share_assumptions"] = counts.n_assumptions / counts.assumptions
    counts["share_responses"] = counts.n_responses / counts.responses

    counts["control_share"] = float("nan")
    counts["lift"] = float("nan")
    if by == "persona_type":
        control = counts[counts[by] == NO_PERSONA]
        if not control.empty:
            keys = [*dimensions, "topic"]
            base = control.set_index(keys).share_assumptions
            lookup = pd.MultiIndex.from_frame(counts[keys])
            reference = base.reindex(lookup).to_numpy()
            counts["control_share"] = reference
            counts["lift"] = _ratio(counts.share_assumptions.to_numpy(), reference)
    return counts.sort_values([*dimensions, by, "share_assumptions"],
                              ascending=[True] * (len(dimensions) + 1) + [False])


def topic_entropy(shares: pd.DataFrame, by: str = "persona_type") -> pd.DataFrame:
    """How spread out each group's topic distribution is, in bits.

    A facet that concentrates the model on one topic is doing something
    different from one that shifts which topic it lands on; entropy separates
    those two the same way the confidence table does for the probe's own
    probabilities.
    """
    if shares.empty:
        return pd.DataFrame()
    dimensions = [c for c in model_dimensions(shares)]
    group = [*dimensions, by]

    def _bits(series) -> float:
        values = [p for p in series if p and p > 0]
        return -sum(p * math.log(p, 2) for p in values) if values else float("nan")

    return (shares.groupby(group, dropna=False, observed=True)
            .agg(topics=("topic", "nunique"),
                 assumptions=("n_assumptions", "sum"),
                 entropy_bits=("share_assumptions", _bits))
            .round(3).reset_index())


# ---------------------------------------------------------------------------
# LLM topic labels
# ---------------------------------------------------------------------------
LABEL_SYSTEM = (
    "You name topics from a topic model. Reply with the label only: a short "
    "noun phrase of at most five words, no quotes, no punctuation, no "
    "explanation."
)

LABEL_TEMPLATE = """\
These are assumptions a language model verbalized about the person it was \
talking to. A topic model grouped some of them together.

Top words for this topic: {words}

Example assumptions in this topic:
{examples}

Name this topic."""


def label_topics(info: pd.DataFrame, model_alias: str, *,
                 dry_run: bool = False, max_words: int = 5) -> pd.DataFrame:
    """Label each topic from its top words, the paper's step with GPT-4o.

    Any alias in `config/models.yaml` works -- the labeler is a model choice,
    not a fixed one, and recording which model wrote the labels matters because
    the labels are data. `dry_run` routes to the mock backend so the path is
    exercisable with no key.
    """
    from syco.model_registry import load_registry
    from syco.models import Conversation, build_adapter

    spec = load_registry().get(model_alias)
    out = info.copy()
    labels = []
    with build_adapter(spec, dry_run=dry_run) as adapter:
        for _, row in out.iterrows():
            if int(row.topic) < 0:
                labels.append("(outliers)")
                continue
            examples = "\n".join(f"- {e}" for e in
                                 str(row.get("examples", "")).split(" | ") if e)
            prompt = LABEL_TEMPLATE.format(words=row.top_words or "(none)",
                                           examples=examples or "- (none)")
            conv = Conversation(messages=({"role": "user", "content": prompt},),
                                system=LABEL_SYSTEM)
            try:
                text = adapter.chat(conv, n=1)[0]
            except Exception as err:                       # noqa: BLE001
                labels.append(f"(labeling failed: {type(err).__name__})")
                continue
            labels.append(_clean_label(text, max_words))
    out["label"] = labels
    out["label_model"] = model_alias if not dry_run else f"{model_alias} (mock)"
    return out


_DECORATION_RE = re.compile(r"^(?:label|topic|name)\s*[:\-\u2013]\s*", re.IGNORECASE)


def _clean_label(text: str, max_words: int) -> str:
    """First non-empty line, stripped of the decoration models add to a label.

    Peeled rather than matched in one pass: `**Label:** "Seeking validation"`
    interleaves the marker with the emphasis, so removing either one exposes
    more of the other.
    """
    from syco.models import strip_think

    body, _ = strip_think(text or "")
    for line in body.splitlines():
        for _ in range(4):
            stripped = _DECORATION_RE.sub("", line.strip().strip("*_#`\"\u201c\u201d"))
            if stripped == line:
                break
            line = stripped
        line = line.strip().rstrip(".")
        if line:
            return " ".join(line.split()[:max_words])
    return "(no label)"
