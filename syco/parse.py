"""Raw completion -> the mental models the model verbalized, plus its reply.

Parsing is deliberately separate from the run: `syco.store` keeps the raw text
verbatim, so a parser fix is a re-parse rather than a re-run over a GPU-week of
generations.

Models do not honor the requested format exactly. In practice the failures are:
a ```json fence around the block, prose before it, a trailing comma, a
probability written as "0.4" or 40%, `RESPONSE` spelled as a markdown heading,
and truncation partway through the reply. Each is handled here, and each is
also *reported* -- `parse_status` distinguishes a clean parse from a salvaged
one, because a finding that only holds on salvaged rows is a finding about the
parser.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

CLEAN = "clean"            # the requested JSON parsed as-is
REPAIRED = "repaired"      # parsed after fixing fences/trailing commas
SALVAGED = "salvaged"      # field-by-field regex extraction
FAILED = "failed"          # no mental models found at all

# `RESPONSE:` as its own line, optionally decorated as a markdown heading or
# bolded, which is how instruct models tend to render a requested heading.
_RESPONSE_RE = re.compile(
    r"^[ \t]*(?:#{1,6}[ \t]*)?(?:\*\*|__)?RESPONSE(?:\*\*|__)?[ \t]*:?[ \t]*(?:\*\*|__)?[ \t]*$",
    re.IGNORECASE | re.MULTILINE)
# The same heading inline, e.g. `RESPONSE: You are not wrong...`
# ...and the same heading inline, e.g. `RESPONSE: You are not wrong...`. The
# closing `**` may sit on either side of the colon, since models bold either the
# word or the whole heading.
_RESPONSE_INLINE_RE = re.compile(
    r"(?:^|\n)[ \t]*(?:\*\*|__)?RESPONSE(?:\*\*|__)?[ \t]*:[ \t]*(?:\*\*|__)?[ \t]*",
    re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")
_PERCENT_RE = re.compile(r"^\s*([0-9.]+)\s*%\s*$")

# Non-greedy bodies anchored on the NEXT field name rather than on a balanced
# quote: the descriptions that break the JSON parser are exactly the ones with
# unescaped quotes inside them, and a character class would drop those entries.
_ENTRY_RE = re.compile(
    r"""["']model_name["']\s*:\s*["'](?P<name>.*?)["']\s*,?\s*"""
    r"""(?:["']description["']\s*:\s*["'](?P<desc>.*?)["']\s*,?\s*)?"""
    r"""["']probability["']\s*:\s*(?P<prob>[0-9.eE+-]+|["'][^"']*["'])""",
    re.DOTALL,
)


@dataclass
class MentalModel:
    """One verbalized assumption about the user.

    The probe asks the model to emit these under the JSON key `model_name` --
    the paper's schema, kept verbatim so the instrument is unchanged. The field
    is called `assumption` here because "model" is already taken twice over in
    this codebase (the LLM, and `model_id` on every row), and a column named
    `model_name` next to `model_id` reads as the LLM's name rather than what it
    actually is: the label the LLM gave one hypothesis about its user.
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


def _find_json_object(text: str, needle: str = "mental_models") -> Optional[str]:
    """The smallest balanced {...} containing `needle`, ignoring braces inside
    strings. Brace matching rather than a regex, because descriptions contain
    both braces and quotes."""
    anchor = text.find(needle)
    if anchor < 0:
        return None
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
        name = str(entry.get("model_name") or entry.get("name") or "").strip()
        desc = str(entry.get("description") or "").strip()
        if not name and not desc:
            continue
        out.append(MentalModel(rank=len(out), assumption=name, description=desc,
                               probability=_to_float(entry.get("probability"))))
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


def parse_completion(raw: str) -> ParsedProbe:
    """Pull the verbalized assumptions and the reply out of one completion."""
    if not raw or not raw.strip():
        return ParsedProbe(status=FAILED, notes="empty completion")

    head, response, has_response = _split_response(raw)
    result = ParsedProbe(response=response, has_response=has_response)

    # The block is normally in the head; a model that answered first still put
    # it somewhere, so fall back to the whole completion.
    for source, salvage_note in ((head, ""), (raw, "block after RESPONSE")):
        block = _find_json_object(source)
        if block is None:
            fenced = _FENCE_RE.search(source)
            block = fenced.group(1).strip() if fenced else None
        if block is None:
            continue

        for candidate, status in ((block, CLEAN),
                                  (_TRAILING_COMMA_RE.sub(r"\1", block), REPAIRED)):
            try:
                entries = _entries_from_obj(json.loads(candidate, strict=False))
            except Exception:
                continue
            models = _build(entries)
            if models:
                result.status = status
                result.mental_models = models
                result.notes = salvage_note
                break
        if result.mental_models:
            break

    if not result.mental_models:
        # Field-by-field: survives truncation mid-block and unescaped quotes
        # inside a description, which is what actually breaks these.
        models = [
            MentalModel(rank=i, assumption=m.group("name").strip(),
                        description=(m.group("desc") or "").strip(),
                        probability=_to_float(m.group("prob")))
            for i, m in enumerate(_ENTRY_RE.finditer(raw))
        ]
        if models:
            result.status = SALVAGED
            result.mental_models = models
        else:
            result.status = FAILED
            result.notes = "no mental_models block"

    probs = [m.probability for m in result.mental_models if m.probability is not None]
    result.prob_sum = round(sum(probs), 6) if probs else None

    if result.mental_models and not has_response:
        # Truncated before the heading, or the model skipped it. Either way the
        # reply is missing, which matters when comparing to the answers table.
        result.notes = (result.notes + "; " if result.notes else "") + "no RESPONSE heading"
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


# What one results row carries by default: the design cell, which model produced
# it, the assumption itself, and the one flag that says whether the row can be
# trusted. `rep` is in here rather than behind --full on purpose -- drop it and
# a multi-draw run silently looks like duplicate rows.
LEAN_COLUMNS = (
    "model_id", "persona_type", "persona_id", "prompt_type", "prompt_id", "rep",
    "rank", "assumption", "description", "probability", "probability_norm",
    "parse_status",
)


def to_records(row: dict, parsed: ParsedProbe, full: bool = False) -> list:
    """One tidy record per verbalized assumption, with the cell's design columns.

    A cell whose parse failed still yields one record with `rank = None`, so a
    failure is a visible row rather than a silently missing cell.

    `full=True` adds the provenance and per-cell diagnostics. They are off by
    default because they are constant across a run -- every row would repeat the
    same backend and probe label -- and the JSONL already holds them per cell.
    """
    record = {
        "cell_key": row.get("cell_key"),
        "model_id": row.get("model_id"),
        "persona_type": row.get("persona_type"),
        "persona_id": row.get("persona_id"),
        "prompt_type": row.get("prompt_type"),
        "prompt_id": row.get("prompt_id"),
        "rep": row.get("rep"),
        "probe": row.get("probe"),
        "history_mode": row.get("history_mode"),
        "backend": row.get("backend"),
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
