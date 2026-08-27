#!/usr/bin/env python3
"""What the verbalized assumptions are about: words, bigrams, and topics.

    python -m syco topics results/gemma3-12b_openended_assumptions.parquet

`summarize_assumptions.py` counts assumption labels as exact strings, so
"wants validation" and "seeking validation" are two different rows. This is the
content pass that looks through the wording, following Cheng et al.,
*Verbalizing LLMs' assumptions to explain and control sycophancy*: unigram and
bigram frequency, then sentence-transformer embeddings clustered with BERTopic
and each topic labeled by an LLM from its top words.

Six tables:

  1. corpus -- what each frequency below is computed over. A word share is per
     *assumption* and a bigram share is per *response*; the paper quotes each
     against its own denominator and they differ by roughly k, so both are on
     every row here and this table is where the denominators come from.

  2. words, and 3. bigrams -- the paper's own descriptives. Unigrams drop
     function words; bigrams keep them, because the paper's bigram table
     reports "rather than" and "may have" and a stopword filter destroys those.

  4. terms by persona facet -- which words the model reaches for more when a
     facet is disclosed than with no persona at all. This is the study's
     question rather than the paper's, and the reason lift is on every table.

  5. topics -- BERTopic over the whole input, so the facets share one topic
     space and are comparable within it. Fitting per facet would give each its
     own topics and nothing to compare. Read `outliers` alongside it: those are
     assumptions the model had no coherent topic for, and they are kept rather
     than reassigned unless `--reduce-outliers` says otherwise.

  6. topic share by facet -- the same contrast as 4, over topics instead of
     words, plus the entropy of each facet's topic distribution. A facet that
     concentrates the model on one topic is doing something different from one
     that moves which topic it lands on.

Tables 5 and 6 need bertopic and sentence-transformers; 1-4 need nothing
beyond pandas, and are printed either way.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

# numpy fixes its BLAS pool size when it is first imported, which pandas does
# on the next line -- before argparse has seen --threads. A default set here is
# the only place it can still take effect; an operator who wants the machine's
# full width sets OPENBLAS_NUM_THREADS in the environment and this stands down.
# Every other pool is capped from --threads inside syco.topics.fit_topics.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from syco.data import NO_PERSONA
from syco.tables import load, model_dimensions
from syco.topics import (
    DEFAULT_EMBEDDING_MODEL,
    STOPWORD_MODES,
    TEXT_FIELDS,
    corpus_profile,
    fit_topics,
    label_topics,
    ngram_frequencies,
    topic_entropy,
    topic_shares,
    topics_available,
)

DISPLAY = {
    "n_assumptions": "n", "share_assumptions": "of_assumptions",
    "n_responses": "n_resp", "share_responses": "of_responses",
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="*_assumptions.parquet from parse_assumptions.py")
    p.add_argument("--field", choices=TEXT_FIELDS, default="description",
                   help="which text to analyze. Default `description`: the's "
                        "paper's frequency tables are computed over the "
                        "assumption explanations (Appendix A), not the short "
                        "labels. `assumption` is the label alone, `both` joins "
                        "them as separate phrases")
    p.add_argument("--top", type=int, default=15,
                   help="terms/topics to print per table (default: 15)")
    p.add_argument("--min-count", type=int, default=2,
                   help="drop terms appearing in fewer than N assumptions")
    p.add_argument("--stopwords", choices=sorted(STOPWORD_MODES), default="unigrams",
                   help="which n-gram levels drop function words "
                        "(default: unigrams only, as in the paper)")
    p.add_argument("--by", default="persona_type,prompt_type",
                   help="comma-separated design columns to break terms down by")

    topics = p.add_argument_group("topic model")
    topics.add_argument("--no-topics", action="store_true",
                        help="n-grams only; skip BERTopic even if it is installed")
    topics.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL,
                        help=f"sentence-transformer (default: {DEFAULT_EMBEDDING_MODEL})")
    topics.add_argument("--min-topic-size", type=int, default=10,
                        help="smallest topic HDBSCAN may form (default: 10)")
    topics.add_argument("--nr-topics", default=None,
                        help="reduce to N topics, or 'auto'")
    topics.add_argument("--seed", type=int, default=1000,
                        help="UMAP random_state; without it topics are not "
                             "reproducible between runs (default: 1000)")
    topics.add_argument("--device", default="auto",
                        help="torch device for the embeddings: auto (default) "
                             "takes a GPU with free memory and falls back to "
                             "CPU when the cards are busy; or cpu / cuda / "
                             "cuda:N to decide explicitly")
    topics.add_argument("--threads", type=int, default=4,
                        help="cap on every BLAS/OpenMP pool (default: 4)")
    topics.add_argument("--reduce-outliers", action="store_true",
                        help="reassign topic -1 by embedding similarity. Changes "
                             "what every share below means; off by default")
    topics.add_argument("--label-model", default=None,
                        help="alias from config/models.yaml to name each topic "
                             "from its top words (the paper uses GPT-4o)")
    topics.add_argument("--label-dry-run", action="store_true",
                        help="route --label-model to the mock backend")

    p.add_argument("--out-prefix", default=None,
                   help="output prefix (default: the input minus _assumptions)")
    p.add_argument("--no-write", action="store_true", help="print only")
    p.add_argument("--xlsx", default=None, help="also write every table to an .xlsx")
    return p.parse_args(argv)


def default_prefix(path: str) -> pathlib.Path:
    stem = pathlib.Path(path)
    return stem.with_name(stem.stem.removesuffix("_assumptions"))


def constant_columns(df: pd.DataFrame) -> dict:
    """Experiment-identity columns with one value in the whole input.

    They belong on every written row -- a table that loses its `run_id` cannot
    be pooled safely later -- but repeating a 20-character hash down a terminal
    table buys nothing, so they are printed once in the header instead.
    """
    return {column: df[column].iloc[0] for column in model_dimensions(df)
            if df[column].nunique(dropna=False) == 1}


def trim(table: pd.DataFrame, **limits: int) -> pd.DataFrame:
    """Shorten comma/pipe-joined columns for printing only.

    Ten c-TF-IDF words and three example labels are what the written table
    should carry; printed in full they push a topic table past any terminal.
    """
    table = table.copy()
    for column, keep in limits.items():
        if column not in table.columns:
            continue
        separator = " | " if column == "examples" else ", "
        table[column] = table[column].astype(str).map(
            lambda value, sep=separator, k=keep: sep.join(value.split(sep)[:k]))
    return table


def show(table: pd.DataFrame, drop=()) -> str:
    if table is None or len(table) == 0:
        return "(nothing to show)"
    table = table.drop(columns=[c for c in drop if c in table.columns])
    return table.rename(columns=DISPLAY).to_string(index=False)


def degenerate(info: pd.DataFrame, outlier_share: float) -> list[str]:
    """Ways a fitted topic model can be technically fine and analytically empty.

    A model that puts nine in ten assumptions in one topic has not found
    structure, it has found that the corpus is homogeneous under this
    embedding -- and the per-facet table below it will then show every facet
    looking alike for a reason that has nothing to do with the facets. Same for
    a model that called most of the corpus noise. Neither is an error, so
    neither stops the run; both are worth saying out loud.
    """
    warnings = []
    real = info[info.topic >= 0]
    if len(real) < 2:
        warnings.append(
            "fewer than two topics -- nothing to compare across facets. Try a "
            "smaller --min-topic-size, or --field assumption if this was run "
            "on the longer descriptions.")
    elif real.share_assumptions.max() > 0.8:
        biggest = real.share_assumptions.max()
        warnings.append(
            f"one topic holds {biggest:.0%} of the assumptions, so the "
            "per-facet shares below are close to constant by construction. "
            "A smaller --min-topic-size splits it further.")
    if outlier_share > 0.5:
        warnings.append(
            f"{outlier_share:.0%} of assumptions were left as outliers; the "
            "topics describe a minority of the corpus.")
    return warnings


def _facet_terms(ngrams: pd.DataFrame, top: int) -> tuple[pd.DataFrame, str]:
    """Per-facet terms, ranked by lift over the control where there is one."""
    facet = ngrams[(ngrams.scope == "persona_type") & (ngrams.group != NO_PERSONA)]
    if facet.empty:
        return pd.DataFrame(), "no persona facets in this table"
    has_control = facet.lift.notna().any()
    if has_control:
        ranked = facet[facet.lift.notna()].sort_values("lift", ascending=False)
        note = "ranked by lift over the persona-free control"
    else:
        ranked = facet.sort_values("share_assumptions", ascending=False)
        note = (f"no {NO_PERSONA!r} control cells in this run, so there is no "
                "lift to compute -- ranked by share instead")
    keys = [*model_dimensions(ranked), "level", "group"]
    return (ranked.groupby(keys, dropna=False, observed=True, group_keys=False)
            .head(top), note)


def main(argv=None) -> int:
    args = parse_args(argv)
    df = load(args.input)
    by = tuple(column.strip() for column in args.by.split(",") if column.strip())

    constants = constant_columns(df)
    elide = tuple(constants)
    tables: dict[str, pd.DataFrame] = {}
    tables["corpus"] = corpus_profile(df, args.field)
    ngrams = ngram_frequencies(df, field=args.field, levels=(1, 2),
                               stopword_mode=args.stopwords, by=by,
                               top=max(args.top, 50), min_count=args.min_count)
    tables["ngrams"] = ngrams

    title = f"analysis of {len(df)} verbalized assumption(s) from {args.input}"
    print(f"{title}\n{'=' * len(title)}")
    for column, value in constants.items():
        print(f"  {column:<13} {value}")
    print(f"\n1. corpus (field={args.field}, stopwords={args.stopwords})\n"
          f"{'-' * 40}")
    print(show(tables["corpus"], drop=elide))

    overall = ngrams[ngrams.scope == "overall"] if not ngrams.empty else pd.DataFrame()
    for number, (level, name) in enumerate((("unigram", "words"),
                                            ("2-gram", "bigrams")), start=2):
        rows = overall[overall.level == level].head(args.top) if len(overall) else None
        header = f"{number}. most frequent {name} (top {args.top})"
        print(f"\n{header}\n{'-' * len(header)}")
        print(show(rows, drop=(*elide, "level", "scope", "group",
                               "control_share", "lift")))

    facet_terms, note = _facet_terms(ngrams, args.top) if not ngrams.empty \
        else (pd.DataFrame(), "no terms")
    tables["terms_by_facet"] = facet_terms
    header = f"4. terms by persona facet (top {args.top} per facet and level)"
    print(f"\n{header}\n{'-' * len(header)}\n({note})")
    print(show(facet_terms, drop=(*elide, "scope", "assumptions", "responses",
                                  "n_responses", "share_responses")))

    rc = _topics(df, args, tables, elide)

    if not args.no_write:
        prefix = pathlib.Path(args.out_prefix) if args.out_prefix \
            else default_prefix(args.input)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        written = []
        for name, suffix in (("ngrams", "_ngrams.parquet"),
                             ("topics", "_topics.parquet"),
                             ("topic_shares", "_topic_shares.parquet"),
                             ("topic_assignments", "_topic_assignments.parquet")):
            table = tables.get(name)
            if table is None or table.empty:
                continue
            path = prefix.with_name(prefix.name + suffix)
            table.to_parquet(path, index=False)
            written.append(str(path))
        print("\nwrote " + ("\n      ".join(written) if written else "nothing"))

    if args.xlsx:
        with pd.ExcelWriter(args.xlsx) as writer:
            for name, table in tables.items():
                if table is not None and not table.empty:
                    table.to_excel(writer, sheet_name=name[:31], index=False)
        print(f"wrote {args.xlsx}")
    return rc


def _topics(df: pd.DataFrame, args, tables: dict, elide=()) -> int:
    """Tables 5 and 6. Returns 0 even when the stack is absent -- an optional
    dependency that is not installed is a skipped table, not a failed run."""
    header = "5. topics (BERTopic)"
    print(f"\n{header}\n{'-' * len(header)}")
    if args.no_topics:
        print("(skipped: --no-topics)")
        return 0
    available, why = topics_available()
    if not available:
        print(f"(skipped: {why})")
        return 0
    try:
        result = fit_topics(
            df, field=args.field, embedding_model=args.embedding_model,
            min_topic_size=args.min_topic_size,
            nr_topics=None if args.nr_topics in (None, "none") else (
                args.nr_topics if args.nr_topics == "auto" else int(args.nr_topics)),
            seed=args.seed, reduce_outliers=args.reduce_outliers,
            device=args.device, threads=args.threads,
        )
    except RuntimeError as err:
        print(f"(skipped: {err})")
        return 0

    info = result.info
    if args.label_model:
        info = label_topics(info, args.label_model, dry_run=args.label_dry_run)
    tables["topics"] = info

    assignments = df[[c for c in (
        "run_id", "probe", "persona_type",
        "persona_id", "prompt_type", "prompt_id", "rep", "rank", "assumption",
    ) if c in df.columns]].copy()
    assignments["topic"] = result.assignments
    tables["topic_assignments"] = assignments

    effective = result.params["min_topic_size_effective"]
    clamped = "" if effective == args.min_topic_size else \
        f" (clamped from {args.min_topic_size} for this corpus)"
    print(f"({result.n_topics} topic(s) over {result.params['n_documents']} "
          f"assumptions; embeddings={args.embedding_model} on "
          f"{result.params['device']}, "
          f"seed={args.seed}, "
          f"min_topic_size={effective}{clamped}; "
          f"outliers={result.outlier_share:.1%}"
          f"{', reassigned' if args.reduce_outliers else ', kept as topic -1'})")
    print(show(trim(info.head(args.top), top_words=6, examples=2),
               drop=("n_responses", "share_responses")))
    if args.device == "auto":
        where = result.params["device"]
        print(f"   note: --device auto resolved to {where}. CPU and GPU fit "
              "measurably different partitions from the same seed, so pin "
              f"--device {where} to reproduce this table elsewhere.")
    for warning in degenerate(info, result.outlier_share):
        print(f"   note: {warning}")

    shares = topic_shares(df, result.assignments, by="persona_type")
    tables["topic_shares"] = shares
    header = "6. topic share by persona facet"
    print(f"\n{header}\n{'-' * len(header)}")
    if shares.empty:
        print("(nothing to show)")
        return 0
    # Without --label-model there is no name for a topic, so its leading
    # c-TF-IDF words stand in -- trimmed hard, since this column repeats on
    # every facet row and the full list is in the topics table above.
    named = info.set_index("topic")["label"].to_dict() if "label" in info else \
        trim(info, top_words=3).set_index("topic")["top_words"].to_dict()
    shown = shares.assign(topic_label=shares.topic.map(named))
    print(show(shown.groupby([*model_dimensions(shares), "persona_type"],
                             dropna=False, observed=True, group_keys=False)
               .head(args.top),
               drop=(*elide, "n_responses", "responses", "share_responses")))
    entropy = topic_entropy(shares)
    print("\n   topic-distribution entropy per facet (bits)")
    print(show(entropy, drop=elide))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
