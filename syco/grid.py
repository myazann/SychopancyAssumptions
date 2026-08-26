"""The design: which cells exist, and which subset a run administers.

A cell is one (persona_type, persona_id, prompt_type, prompt_id) combination,
optionally repeated. Fully crossed that is 10 x 200 x 2 x 1000 = 4M cells, so
every run works on a subset -- and *how* the subset is drawn decides which
questions the data can answer.

The rule here is that subsetting is always PAIRED: the same persona people and
the same dilemmas appear in every condition. Draw personas and prompts once,
then cross them with all the levels. That makes both contrasts within-subject:

  * persona-trait bias -- the same person, the same dilemma, ten different
    facets of them disclosed. Any difference is attributable to the facet.
  * framing/sycophancy -- the same person, the same dilemma, told from either
    side. `original_post` and `flipped_story` share a `prompt_id`, so they
    always enter or leave the sample together.

Sampling independently per condition would confound the contrast with whichever
personas and dilemmas happened to be drawn, and no amount of downstream
analysis recovers from that.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Optional

from syco.data import CONTROL_PERSONA, FLIPPED, NO_PERSONA, ORIGINAL, Persona, Prompt


@dataclass(frozen=True)
class Cell:
    persona: Persona
    prompt: Prompt
    rep: int = 0                 # repeat index, for multi-draw cells

    @property
    def key_parts(self) -> tuple:
        return (self.persona.persona_type, self.persona.persona_id,
                self.prompt.prompt_type, self.prompt.prompt_id, self.rep)

    def describe(self) -> str:
        return (f"{self.persona.persona_type}/{self.persona.persona_id[:8]} "
                f"x {self.prompt.prompt_type}/{self.prompt.prompt_id} #{self.rep}")


def cell_key(model_alias: str, probe_label: str, cell: Cell,
             run_id: str = "") -> str:
    """Stable identity of a cell -- the unit of resume.

    Includes the probe label, so switching `--history-mode` or the number of
    mental models starts a fresh set of cells in the same file instead of
    silently resuming rows collected under a different instrument.
    """
    parts = (model_alias, probe_label, *cell.key_parts)
    if run_id:
        parts = (run_id, *parts)
    return "|".join(str(p) for p in parts)


def _stable_sample(items: list, n: Optional[int], seed: int, tag: str) -> list:
    """A deterministic subset, drawn the same way on every re-run.

    Seeded off `tag` as well as `seed`, so the persona draw and the prompt draw
    are independent even at the same seed -- otherwise they would be correlated
    in whatever order the tables happen to be in.
    """
    if n is None or n >= len(items):
        return list(items)
    digest = hashlib.sha256(f"{seed}|{tag}".encode()).hexdigest()
    rng = random.Random(int(digest, 16) % (2**32))
    return rng.sample(list(items), n)


def build_cells(
    personas: list,
    prompts: list,
    *,
    persona_types: Optional[list] = None,
    prompt_types: Optional[list] = None,
    n_persona_ids: Optional[int] = None,
    n_prompt_ids: Optional[int] = None,
    include_no_persona: bool = True,
    n_reps: int = 1,
    seed: int = 1000,
    restrict_pairs: Optional[set] = None,
    restrict_cells: Optional[set] = None,
) -> list:
    """Expand the design. Pure -- makes no model calls, so a grid can be planned
    and counted before anything is loaded.

    `restrict_cells` is a set of
    (persona_type, persona_id, prompt_type, prompt_id) coordinates that the run
    is limited to. `--match-existing` fills it from a prior answers table so an
    assumption row cannot be generated for a facet or framing that has no
    corresponding existing answer.

    `restrict_pairs` is retained for compatibility with older callers. It has
    the weaker (persona_id, prompt_id) semantics and should not be used for new
    matching workflows.
    """
    if restrict_pairs is not None and restrict_cells is not None:
        raise ValueError("pass only one of restrict_pairs or restrict_cells")
    by_type: dict = {}
    for p in personas:
        by_type.setdefault(p.persona_type, {})[p.persona_id] = p

    # Preserve the source table's facet order. Sorting put the legitimate
    # `assumptions` facet first and made a partial run look as if every persona
    # had been mislabeled as that one facet.
    types = [t for t in (persona_types or list(by_type)) if t in by_type]
    if persona_types:
        missing = [t for t in persona_types if t not in by_type]
        if missing:
            raise ValueError(f"Unknown persona_type(s): {missing}. "
                             f"Known: {sorted(by_type)}")

    # Only people present in EVERY selected facet are eligible, so the
    # trait contrast never compares different sets of people.
    complete = set.intersection(*(set(by_type[t]) for t in types)) if types else set()
    if restrict_cells is not None:
        complete &= {pid for _, pid, _, _ in restrict_cells if pid != NO_PERSONA}
    elif restrict_pairs is not None:
        complete &= {pid for pid, _ in restrict_pairs}
    persona_ids = sorted(complete)
    persona_ids = _stable_sample(persona_ids, n_persona_ids, seed, "persona")

    # Same for dilemmas: a prompt_id is eligible only if both framings exist,
    # so original/flipped enter and leave the sample together.
    by_prompt: dict = {}
    for q in prompts:
        by_prompt.setdefault(q.prompt_id, {})[q.prompt_type] = q

    want_framings = [t for t in (prompt_types or (ORIGINAL, FLIPPED))]
    eligible = [pid for pid, framings in by_prompt.items()
                if all(f in framings for f in want_framings)]
    if restrict_cells is not None:
        eligible = [pid for pid in eligible if pid in {q for _, _, _, q in restrict_cells}]
    elif restrict_pairs is not None:
        eligible = [pid for pid in eligible if pid in {q for _, q in restrict_pairs}]
    prompt_ids = _stable_sample(sorted(eligible), n_prompt_ids, seed, "prompt")

    cells = []
    # Interleave facets within each person/dilemma instead of writing one giant
    # facet block at a time. The full grid is unchanged, but a pilot, --limit
    # run, or snapshot of an in-progress run now covers all persona facets.
    persona_groups = []
    if include_no_persona:
        persona_groups.append([CONTROL_PERSONA])
    persona_groups.extend(
        [by_type[t][pid] for t in types]
        for pid in persona_ids
    )

    for prompt_id in prompt_ids:
        for framing in want_framings:
            for personas_for_cell in persona_groups:
                for persona in personas_for_cell:
                    if restrict_cells is not None:
                        coordinate = (
                            persona.persona_type,
                            persona.persona_id,
                            framing,
                            prompt_id,
                        )
                        if coordinate not in restrict_cells:
                            continue
                    elif restrict_pairs is not None and not persona.is_control:
                        if (persona.persona_id, prompt_id) not in restrict_pairs:
                            continue
                    prompt = by_prompt[prompt_id][framing]
                    for rep in range(n_reps):
                        cells.append(Cell(persona=persona, prompt=prompt, rep=rep))
    return cells


def pairs_from_answers(df, persona_types: Optional[list] = None) -> set:
    """(persona_id, prompt_id) pairs a prior answers table already covers."""
    sub = df
    if persona_types:
        sub = sub[sub["persona_type"].isin(list(persona_types) + [NO_PERSONA])]
    return {(str(a), str(b)) for a, b in zip(sub["persona_id"], sub["prompt_id"])
            if str(a) != NO_PERSONA}


def cells_from_answers(df, persona_types: Optional[list] = None) -> set:
    """Full design coordinates covered by a prior answers table."""
    required = {"persona_type", "persona_id", "prompt_type", "prompt_id"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Existing answers table is missing columns: {missing}")
    sub = df
    if persona_types:
        sub = sub[sub["persona_type"].isin(list(persona_types) + [NO_PERSONA])]
    return {
        (str(ptype), str(pid), str(qtype), str(qid))
        for ptype, pid, qtype, qid in zip(
            sub["persona_type"], sub["persona_id"],
            sub["prompt_type"], sub["prompt_id"],
        )
    }


def summarize(cells: list) -> str:
    from collections import Counter

    by_type = Counter(c.persona.persona_type for c in cells)
    by_framing = Counter(c.prompt.prompt_type for c in cells)
    people = {c.persona.persona_id for c in cells} - {NO_PERSONA}
    dilemmas = {c.prompt.prompt_id for c in cells}
    lines = [
        f"{len(cells)} cells: {len(people)} people x {len(by_type)} persona facet(s) "
        f"x {len(dilemmas)} dilemma(s) x {len(by_framing)} framing(s)",
        "  facets:   " + ", ".join(f"{k}={v}" for k, v in sorted(by_type.items())),
        "  framings: " + ", ".join(f"{k}={v}" for k, v in sorted(by_framing.items())),
    ]
    return "\n".join(lines)
