"""Raw completion -> the mental models the model verbalized, plus its reply.

Parsing is deliberately separate from the run: `syco.store` keeps the raw text
verbatim, so a parser fix is a re-parse rather than a re-run over a GPU-week of
generations.

Models do not honor the requested format exactly. In practice the failures are:
a ```json fence around the block, prose before it, a trailing comma, a
probability written as "0.4" or 40%, `RESPONSE` spelled as a markdown heading,
alternate field names, a numbered Markdown field list, and truncation partway
through the reply. Each is handled here, and each is also *reported* --
`parse_status` distinguishes a clean parse from a salvaged one, because a
finding that only holds on salvaged rows is a finding about the parser.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

CLEAN = "clean"            # the requested JSON parsed as-is
REPAIRED = "repaired"      # parsed after fixing fences/trailing commas
SALVAGED = "salvaged"      # field-by-field regex extraction
INVALID_ORDER = "invalid_order"  # assumptions appeared only after the reply
FAILED = "failed"          # no mental models found at all

# `RESPONSE:` (or a common model-generated synonym) as its own line,
# optionally decorated as a markdown heading or bolded.
_RESPONSE_RE = re.compile(
    r"^[ \t]*(?:#{1,6}[ \t]*)?(?:\*\*|__)?(?:ACTUAL[ \t]+)?(?:RESPONSE|ANSWER)"
    r"(?:\*\*|__)?[ \t]*:?[ \t]*(?:\*\*|__)?[ \t]*$",
    re.IGNORECASE | re.MULTILINE)
# The same heading inline, e.g. `RESPONSE: You are not wrong...`
# ...and the same heading inline, e.g. `RESPONSE: You are not wrong...`. The
# closing `**` may sit on either side of the colon, since models bold either the
# word or the whole heading.
_RESPONSE_INLINE_RE = re.compile(
    r"(?:^|\n)[ \t]*(?:\*\*|__)?(?:ACTUAL[ \t]+)?(?:RESPONSE|ANSWER)"
    r"(?:\*\*|__)?[ \t]*:[ \t]*(?:\*\*|__)?[ \t]*",
    re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")
_PERCENT_RE = re.compile(r"^\s*([0-9.]+)\s*%\s*$")

# Non-greedy bodies anchored on the NEXT field name rather than on a balanced
# quote: the descriptions that break the JSON parser are exactly the ones with
# unescaped quotes inside them, and a character class would drop those entries.
_ENTRY_RE = re.compile(
    r"""["'](?:model_name|name|assumption|label)["']\s*:\s*["'](?P<name>.*?)["']\s*,?\s*"""
    r"""(?:["']description["']\s*:\s*["'](?P<desc>.*?)["']\s*,?\s*)?"""
    r"""["'](?:probability|confidence|prob)["']\s*:\s*(?P<prob>[0-9.eE+%\-]+|["'][^"']*["'])""",
    re.DOTALL | re.IGNORECASE,
)

# Last-resort support for a numbered/Markdown field list. Requiring all three
# labels avoids turning probability-related prose in the actual reply into an
# assumption.
_LABELED_ENTRY_RE = re.compile(
    r"""(?im)^[ \t]*(?:[-*][ \t]+|\d+[.)][ \t]+)?(?:\*\*|__)?"""
    r"""(?:model[ _]name|name|assumption|label)(?:\*\*|__)?[ \t]*:[ \t]*"""
    r"""(?:\*\*|__)?[ \t]*"""
    r"""(?P<name>[^\n]+?)\s*$\s*"""
    r"""^[ \t]*(?:[-*][ \t]+)?(?:\*\*|__)?description(?:\*\*|__)?"""
    r"""[ \t]*:[ \t]*(?:\*\*|__)?[ \t]*(?P<desc>[^\n]+?)\s*$\s*"""
    r"""^[ \t]*(?:[-*][ \t]+)?(?:\*\*|__)?(?:probability|confidence|prob)"""
    r"""(?:\*\*|__)?[ \t]*:[ \t]*(?:\*\*|__)?[ \t]*"""
    r"""(?P<prob>[0-9.eE+%\-]+|["'][^"']*["'])"""
)


@dataclass
class MentalModel:
    """One verbalized assumption about the user.

    The probe asks the model to emit these under the JSON key `model_name` --
    the paper's schema, kept verbatim so the instrument is unchanged. The field
    is called `assumption` here because "model" already refers to the LLM, while
    this value is the label the LLM gave one hypothesis about its user.
    """
    rank: int
    assumption: str
    description: str
    probability: Optional[float]


@dataclass
class ParsedProbe:
    status: str = FAILED
    mental_models: list = field(default_factory=list)
    response: str = ""
    has_response: bool = False
    prob_sum: Optional[float] = None
    notes: str = ""

    @property
    def n_assumptions(self) -> int:
        return len(self.mental_models)


# ---------------------------------------------------------------------------
def _to_float(value) -> Optional[float]:
    """A probability, however the model chose to write it."""
    if isinstance(value, (int, float)):
        num = float(value)
    else:
        text = str(value).strip().strip("\"'")
        pct = _PERCENT_RE.match(text)
        try:
            num = float(pct.group(1)) / 100.0 if pct else float(text)
        except ValueError:
            return None
    # A model that writes 40 for "40%" is common enough to be worth reading,
    # and a genuine probability is never above 1.
    if 1.0 < num <= 100.0:
        num /= 100.0
    return num if 0.0 <= num <= 1.0 else None


#: Anchor keys for each probe family. The open-ended probe emits a
#: `mental_models` *list*; the structured probes emit a `mental_model` *object*.
#: One character apart, so they get separate patterns rather than one loose one.
_OPEN_ENDED_KEY = r'''["']\s*(?:mental[_ ]?models|mentalModels|models)\s*["']'''
_STRUCTURED_KEY = r'''["']\s*(?:mental[_ ]?model|mentalModel|beliefs|support[_ ]?seeking)\s*["']'''


def _find_json_object(text: str, key_pattern: str = _OPEN_ENDED_KEY) -> Optional[str]:
    """The smallest balanced {...} containing a known key, ignoring braces inside
    strings. Brace matching rather than a regex, because descriptions contain
    both braces and quotes."""
    key = re.search(key_pattern, text, re.IGNORECASE)
    if key is None:
        return None
    anchor = key.start()
    start = text.rfind("{", 0, anchor)
    while start >= 0:
        depth, in_string, escape = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        start = text.rfind("{", 0, start)      # unbalanced: try an outer brace
    return None


def _entries_from_obj(obj) -> list:
    if isinstance(obj, dict):
        for key in ("mental_models", "mentalModels", "models"):
            if isinstance(obj.get(key), list):
                return obj[key]
        return []
    return obj if isinstance(obj, list) else []


def _build(entries: list) -> list:
    out = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("model_name") or entry.get("name") or
                   entry.get("assumption") or entry.get("label") or "").strip()
        desc = str(entry.get("description") or "").strip()
        if not name and not desc:
            continue
        probability = (entry.get("probability") if "probability" in entry else
                       entry.get("confidence", entry.get("prob")))
        out.append(MentalModel(rank=len(out), assumption=name, description=desc,
                               probability=_to_float(probability)))
    return out


def _split_response(text: str) -> tuple:
    """-> (head, response, found). The head is everything before the heading."""
    match = _RESPONSE_RE.search(text)
    if match:
        return text[:match.start()], text[match.end():].strip(), True
    match = _RESPONSE_INLINE_RE.search(text)
    if match:
        return text[:match.start()], text[match.end():].strip(), True
    return text, "", False


def _regex_models(source: str) -> list[MentalModel]:
    matches = list(_ENTRY_RE.finditer(source))
    if not matches:
        matches = list(_LABELED_ENTRY_RE.finditer(source))
    return [
        MentalModel(
            rank=i,
            assumption=m.group("name").strip().strip("`*_ \"'"),
            description=(m.group("desc") or "").strip().strip("`*_ \"'"),
            probability=_to_float(m.group("prob")),
        )
        for i, m in enumerate(matches)
    ]


def parse_completion(raw: str, expected_n: Optional[int] = None) -> ParsedProbe:
    """Pull the verbalized assumptions and the reply out of one completion."""
    if not raw or not raw.strip():
        return ParsedProbe(status=FAILED, notes="empty completion")

    head, response, has_response = _split_response(raw)
    result = ParsedProbe(
        response=response,
        has_response=has_response and bool(response.strip()),
    )

    # The block is normally in the head; a model that answered first still put
    # it somewhere, so fall back to the whole completion.
    for source, salvage_note in ((head, ""), (raw, "block after RESPONSE")):
        block = _find_json_object(source)
        if block is None:
            fenced = _FENCE_RE.search(source)
            block = fenced.group(1).strip() if fenced else None
        if block is None:
            continue

        decorated = source.strip() != block.strip()
        first_status = REPAIRED if decorated else CLEAN
        for candidate, status in ((block, first_status),
                                  (_TRAILING_COMMA_RE.sub(r"\1", block), REPAIRED)):
            try:
                entries = _entries_from_obj(json.loads(candidate, strict=False))
            except Exception:
                continue
            models = _build(entries)
            if models:
                result.status = INVALID_ORDER if salvage_note else status
                result.mental_models = models
                result.notes = salvage_note
                break
        if result.mental_models:
            break

    if not result.mental_models:
        # Field-by-field: survives truncation mid-block and unescaped quotes
        # inside a description, which is what actually breaks these.
        models = _regex_models(raw)
        if models:
            in_head = bool(_regex_models(head))
            result.status = SALVAGED if in_head else INVALID_ORDER
            result.mental_models = models
        else:
            result.status = FAILED
            result.notes = "no mental_models block"

    probs = [m.probability for m in result.mental_models if m.probability is not None]
    result.prob_sum = round(sum(probs), 6) if probs else None

    if result.mental_models and not result.has_response:
        missing = "empty RESPONSE" if has_response else "no RESPONSE heading"
        result.notes = (result.notes + "; " if result.notes else "") + missing
    if (expected_n is not None and result.mental_models and
            result.n_assumptions != expected_n):
        mismatch = f"expected {expected_n} assumptions, found {result.n_assumptions}"
        result.notes = (result.notes + "; " if result.notes else "") + mismatch
    return result


def normalized_probabilities(models: list) -> list:
    """Probabilities renormalized to sum to 1, or None where unusable.

    Models routinely emit 0.5/0.3/0.3. Renormalizing makes the mass comparable
    across cells; `prob_sum` is kept on the row so the raw miscalibration stays
    visible rather than being smoothed away.
    """
    probs = [m.probability for m in models]
    usable = [p for p in probs if p is not None]
    total = sum(usable)
    if not usable or total <= 0:
        return [None] * len(models)
    return [None if p is None else p / total for p in probs]


# What one results row carries by default: the design cell, the assumption
# itself, and the one flag that says whether the row can be
# trusted. `rep` is in here rather than behind --full on purpose -- drop it and
# a multi-draw run silently looks like duplicate rows.
LEAN_COLUMNS = (
    "run_id", "probe", "persona_type", "persona_id",
    "prompt_type", "prompt_id", "rep",
    "rank", "assumption", "description", "probability", "probability_norm",
    "parse_status", "has_response",
)


def to_records(row: dict, parsed: ParsedProbe, full: bool = False) -> list:
    """One tidy record per verbalized assumption, with the cell's design columns.

    A cell whose parse failed still yields one record with `rank = None`, so a
    failure is a visible row rather than a silently missing cell.

    `full=True` adds per-cell diagnostics. Constant model identity and serving
    provenance live in the run manifest rather than either output table.
    """
    record = {
        "cell_key": row.get("cell_key"),
        "run_id": row.get("run_id"),
        "persona_type": row.get("persona_type"),
        "persona_id": row.get("persona_id"),
        "prompt_type": row.get("prompt_type"),
        "prompt_id": row.get("prompt_id"),
        "rep": row.get("rep"),
        "probe": row.get("probe"),
        "persona_turns": row.get("persona_turns"),
        "persona_recovered": row.get("persona_recovered"),
        "parse_status": parsed.status,
        "n_assumptions": parsed.n_assumptions,
        "prob_sum": parsed.prob_sum,
        "has_response": parsed.has_response,
        "response_chars": len(parsed.response),
        "parse_notes": parsed.notes,
    }

    if not parsed.mental_models:
        rows = [dict(record, rank=None, assumption=None, description=None,
                     probability=None, probability_norm=None)]
    else:
        norm = normalized_probabilities(parsed.mental_models)
        rows = [
            dict(record, rank=m.rank, assumption=m.assumption,
                 description=m.description, probability=m.probability,
                 probability_norm=p)
            for m, p in zip(parsed.mental_models, norm)
        ]
    if full:
        return rows
    return [{k: r[k] for k in LEAN_COLUMNS} for r in rows]


# ---------------------------------------------------------------------------
# structured probes: fixed dimensions scored 0-1
# ---------------------------------------------------------------------------
@dataclass
class Belief:
    """One scored dimension from a structured probe."""
    dimension: str
    score: Optional[float]
    explanation: str


@dataclass
class ParsedStructured:
    status: str = FAILED
    beliefs: list = field(default_factory=list)
    response: str = ""
    has_response: bool = False
    notes: str = ""

    @property
    def n_dimensions(self) -> int:
        return len(self.beliefs)


def parse_structured(raw: str, kind: str) -> ParsedStructured:
    """Pull `mental_model.<container>.<dimension>.{score, explanation}` out.

    Unlike the open-ended probe there is no `sums to 1` invariant to check --
    `supporttypes` states outright that its dimensions are independent and do
    NOT need to sum to 1, so nothing here normalizes them. Missing dimensions
    are emitted with a null score rather than dropped, so a cell always has one
    row per dimension the prompt asked for and a gap is visible as a null.
    """
    from syco.prompts import STRUCTURED_CONTAINER, STRUCTURED_DIMENSIONS

    if kind not in STRUCTURED_DIMENSIONS:
        raise ValueError(f"{kind!r} is not a structured probe")
    wanted = STRUCTURED_DIMENSIONS[kind]
    container = STRUCTURED_CONTAINER[kind]

    if not raw or not raw.strip():
        return ParsedStructured(status=FAILED, notes="empty completion")

    head, response, has_response = _split_response(raw)
    result = ParsedStructured(
        response=response,
        has_response=has_response and bool(response.strip()),
    )

    notes = []
    payload = None
    for source, note in ((head, ""), (raw, "block after RESPONSE")):
        block = _find_json_object(source, _STRUCTURED_KEY)
        if block is None:
            fenced = _FENCE_RE.search(source)
            block = fenced.group(1).strip() if fenced else None
        if block is None:
            continue
        for candidate, repair in ((block, ""),
                                  (_TRAILING_COMMA_RE.sub(r"\1", block), "trailing comma")):
            try:
                payload = json.loads(candidate)
            except Exception:
                continue
            if repair:
                notes.append(repair)
            break
        if payload is not None:
            if note:
                notes.append(note)
            break

    if payload is None:
        result.status = FAILED
        result.notes = "; ".join(notes + ["no JSON object found"])
        return result

    # The prompt nests under mental_model -> container, but a model that
    # flattened one level is still readable; take the first dict that holds the
    # dimensions rather than failing the cell over nesting.
    scope = payload
    if isinstance(scope, dict) and "mental_model" in scope:
        scope = scope["mental_model"]
    if isinstance(scope, dict) and container in scope:
        scope = scope[container]
    elif isinstance(scope, dict) and not any(d in scope for d in wanted):
        for value in scope.values():
            if isinstance(value, dict) and any(d in value for d in wanted):
                scope = value
                notes.append("dimensions found under an unexpected key")
                break

    if not isinstance(scope, dict):
        result.status = FAILED
        result.notes = "; ".join(notes + ["no dimension object found"])
        return result

    found = 0
    for dimension in wanted:
        entry = scope.get(dimension)
        if isinstance(entry, dict):
            score = _to_float(entry.get("score"))
            explanation = str(entry.get("explanation") or "").strip()
        elif entry is None:
            score, explanation = None, ""
        else:
            # A model that emitted a bare number instead of {score, explanation}.
            score, explanation = _to_float(entry), ""
            if score is not None:
                notes.append(f"{dimension}: bare score")
        if score is not None:
            found += 1
        result.beliefs.append(Belief(dimension, score, explanation))

    missing = [b.dimension for b in result.beliefs if b.score is None]
    if missing:
        notes.append(f"missing score(s): {', '.join(missing)}")
    out_of_range = [b.dimension for b in result.beliefs
                    if b.score is not None and not 0.0 <= b.score <= 1.0]
    if out_of_range:
        notes.append(f"score(s) outside 0-1: {', '.join(out_of_range)}")

    if found == 0:
        result.status = FAILED
    elif missing or out_of_range:
        result.status = SALVAGED
    elif notes:
        result.status = REPAIRED
    else:
        result.status = CLEAN
    result.notes = "; ".join(notes)
    return result


def to_structured_records(row: dict, parsed: ParsedStructured) -> list:
    """One tidy record per scored dimension, with the cell's design columns."""
    record = {
        "cell_key": row.get("cell_key"),
        "run_id": row.get("run_id"),
        "persona_type": row.get("persona_type"),
        "persona_id": row.get("persona_id"),
        "prompt_type": row.get("prompt_type"),
        "prompt_id": row.get("prompt_id"),
        "rep": row.get("rep"),
        "probe": row.get("probe"),
        "persona_turns": row.get("persona_turns"),
        "persona_recovered": row.get("persona_recovered"),
        "parse_status": parsed.status,
        "n_dimensions": parsed.n_dimensions,
        "has_response": parsed.has_response,
        "response_chars": len(parsed.response),
        "parse_notes": parsed.notes,
    }
    if not parsed.beliefs:
        return [dict(record, dimension=None, score=None, explanation=None)]
    return [dict(record, dimension=b.dimension, score=b.score,
                 explanation=b.explanation) for b in parsed.beliefs]
