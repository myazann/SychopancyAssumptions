"""Loading the study's base data: personas, prompts, and prior answers.

A persona is a *chat history* -- a JSON list of {"role", "content"} messages in
which a synthetic person tells the assistant something about themselves and the
assistant replies. `persona_id` identifies the person; `persona_type` identifies
which facet of them the history reveals (hobbies, politics, family, ...), so the
same 99-2000 people each appear once per facet. That crossing is what makes the
persona-trait contrast within-subject.

A prompt is an AITA-style dilemma with two framings of the same underlying
situation: `original_post` (told by the person who posted it) and
`flipped_story` (the same events retold from the other party's side). Same
`prompt_id`, opposite protagonist -- the sycophancy contrast.

About 4% of `persona_text` values are malformed JSON (truncated strings, a
`"role": "assistant": "content":` typo, a stray `]`). They are LLM-generated
transcripts that were stored as text, so this module salvages what it can with
`recover_messages` and records the rest as unusable rather than dropping them
silently -- a persona that fails to parse is a hole in the design, not noise.
"""
from __future__ import annotations

import ast
import json
import re
import warnings
from dataclasses import dataclass

import pandas as pd

from syco import paths

# Marks the persona-free control cell in every table and cell key.
NO_PERSONA = "none"

ORIGINAL = "original_post"
FLIPPED = "flipped_story"


# ---------------------------------------------------------------------------
# persona transcripts
# ---------------------------------------------------------------------------
# Matches one {"role": "...", "content": "..."} object, tolerating single
# quotes, the `"role": "x": "content":` typo, and a content string that runs to
# the end of the input because it was truncated mid-write.
_MSG_RE = re.compile(
    r"""["']role["']\s*[:,]\s*["'](?P<role>user|assistant|system)["']\s*[:,]\s*"""
    r"""["']content["']\s*:\s*(?P<q>["'])(?P<content>.*?)(?<!\\)(?P=q)\s*[,}\]]""",
    re.DOTALL | re.IGNORECASE,
)
_TAIL_RE = re.compile(
    r"""["']role["']\s*[:,]\s*["'](?P<role>user|assistant|system)["']\s*[:,]\s*"""
    r"""["']content["']\s*:\s*(?P<q>["'])(?P<content>.*)$""",
    re.DOTALL | re.IGNORECASE,
)


def _unescape(raw: str) -> str:
    """Turn a JSON string body back into text without re-parsing the object."""
    try:
        return json.loads(f'"{raw}"', strict=False)
    except Exception:
        return (raw.replace("\\n", "\n").replace("\\t", "\t")
                   .replace('\\"', '"').replace("\\'", "'").replace("\\\\", "\\"))


def _literal_eval_quiet(text: str):
    """Parse Python literals without leaking invalid-escape data warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        return ast.literal_eval(text)


def recover_messages(text) -> list:
    """Best-effort parse of a stored transcript into a message list.

    Three passes, cheapest first: strict-off JSON, Python literal, then a
    regex sweep for message objects. The regex pass is what rescues the
    truncated transcripts -- it keeps every complete message and, if the string
    ends mid-message, keeps that partial content too rather than discarding a
    whole persona over its last few tokens.

    Returns [] for anything unusable, including the empty `[]` control row.
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return []
    text = str(text).strip()
    if not text or text == "[]":
        return []

    for loader in (lambda s: json.loads(s, strict=False), _literal_eval_quiet):
        try:
            parsed = loader(text)
        except Exception:
            continue
        if isinstance(parsed, list):
            out = [
                {"role": str(m["role"]), "content": str(m["content"])}
                for m in parsed
                if isinstance(m, dict) and m.get("role") and m.get("content") is not None
            ]
            if out:
                return out

    out, end = [], 0
    for m in _MSG_RE.finditer(text):
        out.append({"role": m.group("role").lower(), "content": _unescape(m.group("content"))})
        end = m.end()
    tail = _TAIL_RE.search(text[end:])
    if tail:
        content = _unescape(tail.group("content").rstrip('"\'}] \n\t'))
        if content.strip():
            out.append({"role": tail.group("role").lower(), "content": content})
    return out


def normalize_messages(messages: list) -> list:
    """Make a transcript safe to send to any chat API.

    Consecutive same-role turns are merged (79 transcripts have back-to-back
    assistant messages, which the Anthropic API rejects outright), empty turns
    are dropped, and a leading assistant turn is dropped because a conversation
    that opens on the assistant has no user context to condition on. Roles other
    than user/assistant are dropped -- a system turn inside a persona would
    silently override the run's own system prompt.
    """
    clean = []
    for m in messages:
        role = str(m.get("role", "")).lower()
        content = str(m.get("content", "") or "").strip()
        if role not in ("user", "assistant") or not content:
            continue
        if clean and clean[-1]["role"] == role:
            clean[-1]["content"] += "\n\n" + content
        else:
            clean.append({"role": role, "content": content})
    while clean and clean[0]["role"] == "assistant":
        clean.pop(0)
    return clean


def transcript_to_text(messages: list) -> str:
    """Flatten a transcript to the paper's `User: ... / AI: ...` history block."""
    label = {"user": "User", "assistant": "AI"}
    return "\n".join(f"{label.get(m['role'], m['role'])}: {m['content']}" for m in messages)


# ---------------------------------------------------------------------------
# the tables
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Persona:
    persona_id: str
    persona_type: str
    messages: tuple          # ({"role","content"}, ...) already normalized
    recovered: bool = False  # transcript needed the regex salvage pass
    n_turns: int = 0

    @property
    def is_control(self) -> bool:
        return self.persona_id == NO_PERSONA


@dataclass(frozen=True)
class Prompt:
    prompt_id: str
    prompt_type: str
    text: str


CONTROL_PERSONA = Persona(NO_PERSONA, NO_PERSONA, (), False, 0)


def load_personas(path=None, drop_unparseable: bool = True):
    """-> (list[Persona], DataFrame of per-transcript parse diagnostics).

    The diagnostics frame is returned rather than logged so a run can record how
    much of the design it actually had transcripts for.
    """
    df = pd.read_pickle(path or paths.PERSONA_PATH)
    personas, diag = [], []
    for row in df.itertuples(index=False):
        ptype, pid, text = row.persona_type, row.persona_id, row.persona_text
        if pd.isna(ptype) or pd.isna(pid):
            continue                        # the empty `[]` control row
        strict_ok = True
        try:
            json.loads(str(text), strict=False)
        except Exception:
            strict_ok = False
        messages = normalize_messages(recover_messages(text))
        diag.append({
            "persona_id": str(pid), "persona_type": str(ptype),
            "strict_json": strict_ok, "recovered": (not strict_ok) and bool(messages),
            "n_turns": len(messages), "usable": bool(messages),
        })
        if messages or not drop_unparseable:
            personas.append(Persona(
                persona_id=str(pid), persona_type=str(ptype),
                messages=tuple(messages), recovered=not strict_ok,
                n_turns=len(messages),
            ))
    return personas, pd.DataFrame(diag)


def load_prompts(path=None) -> list:
    df = pd.read_pickle(path or paths.PROMPT_PATH)
    return [
        Prompt(str(r.prompt_id), str(r.prompt_type), str(r.prompt_text))
        for r in df.itertuples(index=False)
        if not pd.isna(r.prompt_text)
    ]


def load_answers(path) -> pd.DataFrame:
    """A prior answers table (e.g. `gemma-3-12b-it_long_results.pkl`).

    Its persona/prompt columns are the design cells that model was already run
    on; `--match-existing` reuses them so every assumption row has an answer to
    sit beside. NaN persona columns are the persona-free control.
    """
    df = pd.read_pickle(path)
    df = df.copy()
    for col in ("persona_type", "persona_id"):
        if col in df.columns:
            df[col] = df[col].astype(object).where(df[col].notna(), NO_PERSONA)
    return df
