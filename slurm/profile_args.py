"""Expand an experiment profile into `syco run` arguments, for the sbatch scripts.

`python -m syco run --model X` does NOT read config/experiments/*.yaml -- it goes
straight to scripts/run_assumptions.py, where --n-personas and --n-prompts
default to None, and `_stable_sample` treats None as "take everything". A bare
`syco run --model X` therefore administers the full 200 x 1000 x 11 x 2 grid,
not the profile's design. `run --all` only avoids that because it expands
`profile.run_args(spec)` into explicit flags first; this does the same for a
single-model SLURM job.

Deriving the flags rather than writing them into the sbatch keeps the profile as
the single source of truth. Hardcoding them is what lets a design change land in
the YAML while the job silently keeps running the old one.

    python slurm/profile_args.py --models default
    python slurm/profile_args.py --args   default Gemma3-12B   # NUL-separated

This file lives in slurm/ deliberately: `_source_digest` in syco/manifest.py
hashes syco/, scripts/ and config/, so a helper in any of those would change
every run_id. slurm/ is outside that set.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from syco.experiments import load_profile          # noqa: E402
from syco.model_registry import load_registry      # noqa: E402


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    mode, profile_name = argv[0], argv[1]
    profile = load_profile(profile_name)
    registry = load_registry()
    specs = profile.select_models(registry)

    if mode == "--models":
        for spec in specs:
            print(spec.alias)
        return 0

    if mode == "--args":
        if len(argv) < 3:
            print("--args needs a model alias", file=sys.stderr)
            return 2
        alias = argv[2]
        spec = next((s for s in specs if s.alias == alias), None)
        if spec is None:
            print(f"{alias!r} is not in profile {profile_name!r}: "
                  f"{', '.join(s.alias for s in specs)}", file=sys.stderr)
            return 2
        # NUL-separated: `--system ''` is a legitimately empty argument and must
        # survive the trip through the shell.
        for arg in profile.run_args(spec):
            sys.stdout.write(arg + "\0")
        return 0

    print(f"unknown mode {mode!r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
