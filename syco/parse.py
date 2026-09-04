"""Raw completion -> the mental models the model verbalized, plus its reply.

Parsing is deliberately separate from the run: `syco.store` keeps the raw text
verbatim, so a parser fix is a re-parse rather than a re-run over a GPU-week of
generations.

Models do not honor the requested format exactly. In practice the failures are:
a ```json fence around the block, prose before it, a trailing comma, a
probability written as "0.4" or 40%, `RESPONSE` spelled as a markdown heading,
alternate field names, a numbered Markdown field list, truncation partway
through the reply, a block whose closing delimiters were never written, and an
unescaped `"` inside one explanation. Each is handled here, and each is also
*reported* -- `parse_status` distinguishes a clean parse from a salvaged one,
because a finding that only holds on salvaged rows is a finding about the
parser.

For the 4dims probe, two observed scale deviations are handled explicitly and
reported in `parse_notes`: a negative score is capped at zero, and a response
using a 0-10 scale is rescaled as a whole to 0-1. Supporttypes remains strict
because its prompt already states the required scale.
"""
from __future__ import annotations

import json
import math
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


def _close_unbalanced_object(text: str,
                             key_pattern: str = _OPEN_ENDED_KEY) -> Optional[str]:
    """Re-close a block whose trailing delimiters the model never wrote.

    Llama-3.1-8B ends `mental_model` one brace short on about a quarter of its
    structured completions: every dimension is present and the reply follows
    normally, but the outermost object is never closed, so the balanced scan in
    `_find_json_object` walks off the end and reports nothing at all.

    Truncate at the last delimiter the model did close and reopen nothing --
    just emit the closers it still owed, innermost first. Brackets are tracked
    alongside braces because the open-ended probe nests its entries in a list,
    so a brace-only repair would drop the `]` and produce different invalid
    JSON. Only reached after the balanced scan has already failed, so a
    well-formed block never takes this path, and the result is reported as
    REPAIRED rather than CLEAN.
    """
    key = re.search(key_pattern, text, re.IGNORECASE)
    if key is None:
        return None
    start = text.rfind("{", 0, key.start())
    if start < 0:
        return None
    closer = {"{": "}", "[": "]"}
    stack: list = []
    in_string, escape = False, False
    last_close, owed = None, None
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
        if ch in closer:
            stack.append(ch)
        elif ch in ("}", "]"):
            if not stack or closer[stack[-1]] != ch:
                return None          # crossed delimiters: not a clean truncation
            stack.pop()
            last_close, owed = i, list(stack)
    # An empty `owed` means the block closed itself and something else is wrong
    # with it; only a genuinely unclosed block is repaired here.
    if last_close is None or not owed:
        return None
    return text[start:last_close + 1] + "".join(
        closer[ch] for ch in reversed(owed))


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
            # Same dropped-closing-brace defect as the structured probes. It
            # lands here as REPAIRED rather than falling through to the regex
            # tier below, because re-closing recovers the model's own JSON --
            # probabilities included -- instead of re-deriving it from text.
            block = _close_unbalanced_object(source)
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

    @property
    def n_scored(self) -> int:
        return sum(belief.score is not None for belief in self.beliefs)


STRUCTURED_LEAN_COLUMNS = (
    "run_id", "probe", "persona_type", "persona_id",
    "prompt_type", "prompt_id", "rep",
    "dimension", "score", "explanation", "parse_status", "has_response",
)


def _read_structured_number(value) -> tuple[Optional[float], bool]:
    """Read a finite numeric score and whether it was an explicit percent."""
    if isinstance(value, bool) or value is None:
        return None, False
    was_percent = False
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        text = str(value).strip().strip("\"'")
        percent = _PERCENT_RE.match(text)
        try:
            was_percent = percent is not None
            number = float(percent.group(1)) / 100.0 if percent else float(text)
        except ValueError:
            return None, False
    return (number, was_percent) if math.isfinite(number) else (None, was_percent)


def _normalize_structured_scores(values: list, kind: str) -> tuple[list, list[str]]:
    """Normalize the documented 4dims model deviations as one response scale.

    Values already in [0, 1] stay unchanged. If any non-percent score is above
    one and no score exceeds ten, the response is treated as using a 0-10
    scale and every nonnegative value is divided by ten. Negative 4dims scores
    are capped at zero. Supporttypes retains its strict 0-1 behavior.
    """
    parsed = [_read_structured_number(value) for value in values]
    finite = [number for number, percent in parsed if number is not None and not percent]
    ten_scale = (
        kind == "4dims"
        and any(number > 1.0 for number in finite)
        and all(number <= 10.0 for number in finite)
    )
    normalized = []
    capped = []
    scaled = []
    for index, (number, was_percent) in enumerate(parsed):
        if number is None:
            normalized.append(None)
        elif was_percent:
            normalized.append(number if 0.0 <= number <= 1.0 else None)
        elif kind == "4dims" and number < 0.0:
            normalized.append(0.0)
            capped.append(index)
        elif ten_scale and 0.0 <= number <= 10.0:
            normalized.append(number / 10.0)
            scaled.append(index)
        else:
            normalized.append(number if 0.0 <= number <= 1.0 else None)
    notes = []
    if capped:
        notes.append("negative 4dims score(s) capped at 0")
    if scaled:
        notes.append("4dims score(s) rescaled from 0-10 to 0-1")
    return normalized, notes


def _to_structured_score(value) -> Optional[float]:
    """Backwards-compatible strict 0-1 conversion for one isolated value."""
    number, _ = _read_structured_number(value)
    return number if number is not None and 0.0 <= number <= 1.0 else None


#: `"<dimension>": { ... }` -- just the anchor; the body is read separately,
#: because the entries that break `json.loads` are the ones whose explanation
#: contains an unescaped `"` and so cannot be delimited by quote counting.
_DIMENSION_ANCHOR = r"""["']\s*{0}\s*["']\s*:\s*\{{"""
_SCORE_RE = re.compile(
    r"""["']score["']\s*:\s*(?P<score>[0-9.eE+%\-]+|["'][^"']*["'])""",
    re.IGNORECASE)
# Non-greedy body anchored on the entry's closing brace rather than on a
# balanced quote -- same trick as `_ENTRY_RE`, for the same reason.
_EXPLANATION_RE = re.compile(
    r"""["']explanation["']\s*:\s*["'](?P<explanation>.*?)["']\s*,?\s*\}""",
    re.DOTALL | re.IGNORECASE)


def _regex_beliefs(source: str, wanted: tuple) -> dict:
    """Field-by-field extraction when the block will not parse as JSON.

    The open-ended probe has had this tier since the start (`_regex_models`);
    the structured probes did not, so a single unescaped quote inside one
    explanation failed the whole cell and lost the other three or four
    dimensions with it. Each dimension is read from its own slice, so one bad
    explanation costs only its own entry.
    """
    found = {}
    for dimension in wanted:
        anchor = re.search(_DIMENSION_ANCHOR.format(re.escape(dimension)),
                           source, re.IGNORECASE)
        if anchor is None:
            continue
        # Bound the slice at the next dimension so a missing closing brace
        # cannot let one entry swallow the following ones.
        end = len(source)
        for other in wanted:
            if other == dimension:
                continue
            nxt = re.search(_DIMENSION_ANCHOR.format(re.escape(other)),
                            source[anchor.end():], re.IGNORECASE)
            if nxt is not None:
                end = min(end, anchor.end() + nxt.start())
        slice_ = source[anchor.end():end]
        score = _SCORE_RE.search(slice_)
        explanation = _EXPLANATION_RE.search(slice_)
        if score is None and explanation is None:
            continue
        found[dimension] = (
            score.group("score") if score else None,
            (explanation.group("explanation").strip() if explanation else ""),
        )
    return found


def parse_structured(raw: str, kind: str) -> ParsedStructured:
    """Pull `mental_model.<container>.<dimension>.{score, explanation}` out.

    Unlike the open-ended probe there is no `sums to 1` invariant to check.
    The dimensions remain independent. The only scale normalization is the
    documented 4dims repair in `_normalize_structured_scores`; missing
    dimensions are emitted with a null score rather than dropped.
    """
    from syco.prompts import STRUCTURED_CONTAINER, STRUCTURED_DIMENSIONS

    if kind not in STRUCTURED_DIMENSIONS:
        raise ValueError(f"{kind!r} is not a structured probe")
    wanted = STRUCTURED_DIMENSIONS[kind]
    container = STRUCTURED_CONTAINER[kind]

    if not raw or not raw.strip():
        return ParsedStructured(
            status=FAILED,
            beliefs=[Belief(dimension, None, "") for dimension in wanted],
            notes="empty completion",
        )

    head, response, has_response = _split_response(raw)
    result = ParsedStructured(
        response=response,
        has_response=has_response and bool(response.strip()),
    )

    notes = []
    payload = None
    repaired = False
    after_response = False
    for source, note in ((head, ""), (raw, "block after RESPONSE")):
        block = _find_json_object(source, _STRUCTURED_KEY)
        if block is None:
            fenced = _FENCE_RE.search(source)
            block = fenced.group(1).strip() if fenced else None
        if block is None:
            block = _close_unbalanced_object(source, _STRUCTURED_KEY)
            if block is not None:
                notes.append("unclosed JSON object")
        if block is None:
            continue
        decorated = source.strip() != block.strip()
        for candidate, repair in ((block, ""),
                                  (_TRAILING_COMMA_RE.sub(r"\1", block), "trailing comma")):
            try:
                payload = json.loads(candidate, strict=False)
            except Exception:
                continue
            if repair:
                notes.append(repair)
            repaired = decorated or bool(repair)
            break
        if payload is not None:
            if note:
                notes.append(note)
                after_response = True
            break

    if payload is None:
        # Field-by-field, before giving up: an unescaped `"` in one explanation
        # is not a reason to lose the other dimensions in the cell.
        salvaged = _regex_beliefs(head, wanted) or _regex_beliefs(raw, wanted)
        raw_scores = [salvaged.get(dimension, (None, ""))[0] for dimension in wanted]
        scores, normalization_notes = _normalize_structured_scores(raw_scores, kind)
        if any(score is not None for score in scores):
            result.beliefs = [
                Belief(
                    dimension,
                    scores[index],
                    salvaged.get(dimension, (None, ""))[1],
                )
                for index, dimension in enumerate(wanted)
            ]
            missing = [b.dimension for b in result.beliefs if b.score is None]
            if missing:
                notes.append(f"missing score(s): {', '.join(missing)}")
            notes.extend(normalization_notes)
            notes.append("regex salvage")
            if not result.has_response:
                notes.append("empty RESPONSE" if has_response
                             else "no RESPONSE heading")
            result.status = SALVAGED
            result.notes = "; ".join(notes)
            return result
        result.status = FAILED
        result.beliefs = [Belief(dimension, None, "") for dimension in wanted]
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
        result.beliefs = [Belief(dimension, None, "") for dimension in wanted]
        result.notes = "; ".join(notes + ["no dimension object found"])
        return result

    raw_scores = []
    explanations = []
    for dimension in wanted:
        entry = scope.get(dimension)
        if isinstance(entry, dict):
            raw_scores.append(entry.get("score"))
            explanations.append(str(entry.get("explanation") or "").strip())
        elif entry is None:
            raw_scores.append(None)
            explanations.append("")
        else:
            # A model that emitted a bare number instead of {score, explanation}.
            raw_scores.append(entry)
            explanations.append("")
            notes.append(f"{dimension}: bare score")

    scores, normalization_notes = _normalize_structured_scores(raw_scores, kind)
    notes.extend(normalization_notes)
    invalid = [
        dimension
        for dimension, raw_score, score in zip(wanted, raw_scores, scores)
        if raw_score is not None and score is None
    ]
    result.beliefs = [
        Belief(dimension, score, explanation)
        for dimension, score, explanation in zip(wanted, scores, explanations)
    ]
    found = sum(score is not None for score in scores)

    missing = [b.dimension for b in result.beliefs if b.score is None]
    if missing:
        notes.append(f"missing score(s): {', '.join(missing)}")
    if invalid:
        notes.append(f"invalid score(s): {', '.join(invalid)}")

    if found == 0:
        result.status = FAILED
    elif after_response:
        result.status = INVALID_ORDER
    elif missing:
        result.status = SALVAGED
    elif repaired or notes:
        result.status = REPAIRED
    else:
        result.status = CLEAN
    if result.beliefs and not result.has_response:
        missing_response = "empty RESPONSE" if has_response else "no RESPONSE heading"
        notes.append(missing_response)
    result.notes = "; ".join(notes)
    return result


def to_structured_records(
    row: dict, parsed: ParsedStructured, full: bool = False
) -> list:
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
        "n_dimensions_asked": row.get("n_dimensions_asked"),
        "n_dimensions": parsed.n_dimensions,
        "n_scored": parsed.n_scored,
        "has_response": parsed.has_response,
        "response_chars": len(parsed.response),
        "parse_notes": parsed.notes,
    }
    if not parsed.beliefs:
        rows = [dict(record, dimension=None, score=None, explanation=None)]
    else:
        rows = [dict(record, dimension=b.dimension, score=b.score,
                     explanation=b.explanation) for b in parsed.beliefs]
    if full:
        return rows
    return [{key: result[key] for key in STRUCTURED_LEAN_COLUMNS}
            for result in rows]
