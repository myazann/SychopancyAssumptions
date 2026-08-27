"""The probe must be the paper's probe, not our paraphrase of it.

This diffs `syco.prompts` against the vendored reference implementation by
executing the reference's own prompt builder and comparing output character for
character. It exists because a previous version of `syco/prompts.py` shipped an
invented "native" framing that was never in the paper, and every run made with
it measured a different instrument. A docstring claiming fidelity is not
evidence; this is.
"""
from __future__ import annotations

import pathlib
import re

import pytest

from syco.prompts import (ProbeSpec, build, build_prompt_4dims,
                          build_prompt_openended, build_prompt_supporttypes)

VENDORED = (pathlib.Path(__file__).resolve().parents[1]
            / "verbalizedassumptions" / "verbalized_assumptions"
            / "get_assumptions.py")


def _reference(name: str):
    """Execute one `build_prompt_*` out of the vendored file, without importing
    the module (it pulls torch, google-auth and the OpenAI client at import)."""
    source = VENDORED.read_text()
    match = re.search(rf"def {re.escape(name)}\(.*?\n    return prompt",
                      source, re.DOTALL)
    if match is None:
        pytest.skip(f"{name} not found in {VENDORED}")
    namespace: dict = {}
    exec(match.group(0), namespace)  # noqa: S102 -- vendored reference, read-only
    return namespace[name]


HISTORY = ('User: I spend most weekends restoring furniture.\n'
           'AI: That sounds like satisfying, detailed work.')
POST = ('Dinnertime dilemma\n\nWe had BLTs for dinner. I prepared all of the '
        'ingredients, then made my sandwich and ate it. Was I wrong?')


@pytest.mark.parametrize("history", [HISTORY, "", "   \n  "])
def test_openended_matches_the_paper_character_for_character(history):
    reference = _reference("build_prompt_openended")
    assert build_prompt_openended(history, POST) == reference(history, POST)


def test_empty_history_uses_the_papers_own_placeholder():
    assert "(no previous conversation)" in build_prompt_openended("", POST)


def test_build_sends_one_user_message_with_history_inline():
    """The paper puts the transcript inside the prompt text, never as turns."""
    messages = [{"role": "user", "content": "I restore furniture."},
                {"role": "assistant", "content": "Satisfying work."}]
    out = build(ProbeSpec(), messages, POST)
    assert len(out) == 1 and out[0]["role"] == "user"
    body = out[0]["content"]
    assert "Conversation so far:" in body
    assert "User: I restore furniture." in body
    assert "AI: Satisfying work." in body
    assert "User A now says:" in body


def test_control_cell_has_no_transcript():
    body = build(ProbeSpec(), [], POST)[0]["content"]
    assert "(no previous conversation)" in body


def test_probe_kinds_are_the_papers_own_names():
    source = VENDORED.read_text()
    block = re.search(r"AVAILABLE_PROMPTS = \{(.*?)\}", source, re.DOTALL).group(1)
    theirs = set(re.findall(r'"([^"]+)"\s*:', block)) | set(
        re.findall(r"'([^']+)'\s*:", block))
    from syco.prompts import PROBE_KINDS
    unknown = set(PROBE_KINDS) - theirs
    assert not unknown, f"probe names not in the paper's AVAILABLE_PROMPTS: {unknown}"


@pytest.mark.parametrize("history", [HISTORY, "", "   \n  "])
@pytest.mark.parametrize("name,ours", [
    ("build_prompt_4dims", build_prompt_4dims),
    ("build_prompt_supporttypes", build_prompt_supporttypes),
])
def test_structured_prompts_match_the_paper(name, ours, history):
    assert ours(history, POST) == _reference(name)(history, POST)


def test_structured_dimension_keys_appear_in_their_own_prompt():
    """The parser reads dimensions from STRUCTURED_DIMENSIONS; that list must
    match the JSON keys the prompt actually asks for."""
    from syco.prompts import STRUCTURED_CONTAINER, STRUCTURED_DIMENSIONS
    for kind, builder in (("4dims", build_prompt_4dims),
                          ("supporttypes", build_prompt_supporttypes)):
        text = builder(HISTORY, POST)
        assert f'"{STRUCTURED_CONTAINER[kind]}"' in text
        for dim in STRUCTURED_DIMENSIONS[kind]:
            assert f'"{dim}"' in text, f"{kind}: {dim} not in prompt"


def test_unported_probes_refuse_rather_than_approximate():
    with pytest.raises(NotImplementedError, match="not ported"):
        build(ProbeSpec(kind="twostep"), [], POST)


def test_invented_probe_names_are_rejected():
    with pytest.raises(ValueError, match="unknown probe kind"):
        ProbeSpec(kind="native")
