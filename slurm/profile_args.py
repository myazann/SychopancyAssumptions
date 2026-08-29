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
    python slurm/profile_args.py --ready  openended-extension-45x40  # array indices

This file lives in slurm/ deliberately: it is a job-submission helper, not
part of the acquisition path that `syco.manifest.ACQUISITION_SOURCES` hashes
into every run_id.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# This utility only reads YAML and prints arguments, but importing the profile
# module reaches pandas/numpy through the data-grid helpers.  Login shells can
# have a tight per-session thread allowance; letting OpenBLAS initialize one
# thread per visible CPU has caused this metadata-only command to fail before
# `sbatch` is even called.  It performs no numerical work, so one thread is the
# correct setting here regardless of the caller's environment.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

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

    if mode == "--ready":
        # Which array indices can be submitted right now. A wave is only
        # plannable once every shard it builds on is finished, and under an
        # additive workflow those finish at different times -- so submitting
        # the whole array would queue tasks that cannot start. Ready indices go
        # to stdout; the rest go to stderr with the reason, for the human.
        ready = []
        for index, spec in enumerate(specs):
            try:
                profile.build_cells(spec)
            except Exception as err:  # noqa: BLE001 - reported, never raised
                print(f"  not ready: {spec.alias}: {err}", file=sys.stderr)
                continue
            ready.append(index)
        for index in ready:
            print(index)
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
