"""Unified command line interface for the assumptions study."""
from __future__ import annotations

import argparse
import sys

from syco.experiments import load_profile

HELP = """\
usage: python -m syco <command> [options]

Commands:
  doctor                 validate dependencies, data, profile, and GPUs
  models                 list configured models
  plan                   plan every model in an experiment profile
  smoke                  run mock generation -> parse -> summarize
  run --model MODEL ...  administer one model (existing runner arguments)
  run --all              schedule every profile model across available GPUs
  status                 show successful, missing, error, and attempt counts
  merge                  validate and merge canonical per-model outputs
  parse INPUT ...        parse one open-ended or structured JSONL output
  parse --all            parse every per-model output independently
  summarize INPUT ...    summarize one parsed assumptions/scores table
  summarize --all        summarize every per-model parsed table
  topics INPUT ...       open-ended only: words, bigrams, and topics
  topics --all           the same for every per-model parsed table
  sycophancy STAGE ...   score forced-choice answers and join assumptions
                         (stages: binary, join)
  text STAGE ...         analyze persona or response text descriptively
                         (stages: features, words)
  analyze [options]      the three analyses of the open-ended grid: persona
                         facet and framing, demographics, and sycophancy ->
                         a directory of tables, figures, and findings
  pipeline               run-all, merge, parse-all, summarize-all, topics-all

Profile commands default to config/experiments/default.yaml. Override with:
  --profile NAME_OR_PATH
"""


def _profile_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--profile", default="default",
                        help="profile name or YAML path (default: default)")
    return parser


def _profile_from(args):
    return load_profile(args.profile)


def _models() -> int:
    from syco.model_registry import load_registry

    registry = load_registry()
    header = (f"{'alias':<22} {'family':<8} {'backend':<10} {'quant':<14} "
              f"{'VRAM MiB':<9} {'ref'}")
    print(header)
    print("-" * len(header))
    for spec in registry.select(include_disabled=True):
        flag = "" if spec.enabled else " (disabled)"
        vram = spec.estimated_vram_mib or "-"
        print(
            f"{spec.alias:<22} {spec.family:<8} {spec.backend:<10} "
            f"{spec.quantization.label:<14} {vram!s:<9} {spec.ref}{flag}"
        )
    return 0


def _run_all(argv: list[str]) -> int:
    from syco.orchestrate import run_all

    parser = _profile_parser("Schedule all models in an experiment profile")
    parser.add_argument("--limit-per-model", type=int, default=None,
                        help="run at most N additional cells per model")
    parser.add_argument("--wait-timeout", type=int, default=None,
                        help="fail after N seconds unable to fit a queued model")
    args = parser.parse_args(argv)
    if args.limit_per_model is not None and args.limit_per_model <= 0:
        parser.error("--limit-per-model must be positive")
    if args.wait_timeout is not None and args.wait_timeout <= 0:
        parser.error("--wait-timeout must be positive")
    return run_all(
        _profile_from(args),
        limit_per_model=args.limit_per_model,
        wait_timeout_seconds=args.wait_timeout,
    )


def _parse_all(argv: list[str]) -> int:
    from syco.orchestrate import parse_all

    parser = _profile_parser("Parse every model output")
    args, extra = parser.parse_known_args(argv)
    return parse_all(_profile_from(args), extra)


def _summarize_all(argv: list[str]) -> int:
    from syco.orchestrate import summarize_all

    parser = _profile_parser("Summarize every model output")
    args, extra = parser.parse_known_args(argv)
    return summarize_all(_profile_from(args), extra)


def _topics_all(argv: list[str]) -> int:
    from syco.orchestrate import topics_all

    parser = _profile_parser("Run the content analysis on every model output")
    args, extra = parser.parse_known_args(argv)
    return topics_all(_profile_from(args), extra)


def _pipeline(argv: list[str]) -> int:
    from syco.orchestrate import merge, parse_all, run_all, summarize_all, topics_all

    parser = _profile_parser("Run, validate, parse, and summarize a profile")
    parser.add_argument("--wait-timeout", type=int, default=None)
    args = parser.parse_args(argv)
    profile = _profile_from(args)
    rc = run_all(profile, wait_timeout_seconds=args.wait_timeout)
    if rc:
        return rc
    rc = merge(profile)
    if rc:
        return rc
    rc = parse_all(profile)
    if rc:
        return rc
    rc = summarize_all(profile)
    if rc:
        return rc
    return topics_all(profile)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help", "help"}:
        print(HELP)
        return 0
    command, rest = argv[0], argv[1:]
    try:
        if command == "models":
            return _models()
        if command == "run":
            if "--all" in rest:
                return _run_all([item for item in rest if item != "--all"])
            from scripts.run_assumptions import main as run_main
            return run_main(rest)
        if command == "parse":
            if "--all" in rest:
                return _parse_all([item for item in rest if item != "--all"])
            from scripts.parse_assumptions import main as parse_main
            return parse_main(rest)
        if command == "summarize":
            if "--all" in rest:
                return _summarize_all([item for item in rest if item != "--all"])
            from scripts.summarize_assumptions import main as summarize_main
            return summarize_main(rest)
        if command == "topics":
            if "--all" in rest:
                return _topics_all([item for item in rest if item != "--all"])
            from scripts.topic_assumptions import main as topics_main
            return topics_main(rest)
        if command == "sycophancy":
            from scripts.score_sycophancy import main as sycophancy_main
            return sycophancy_main(rest)
        if command == "text":
            from scripts.analyze_text import main as text_main
            return text_main(rest)
        if command == "analyze":
            from scripts.analyze_openended import main as analyze_main
            return analyze_main(rest)
        if command == "plan":
            from syco.orchestrate import plan
            parser = _profile_parser("Plan every model in a profile")
            args = parser.parse_args(rest)
            return plan(_profile_from(args))
        if command == "status":
            from syco.orchestrate import status
            parser = _profile_parser("Show completion status")
            args = parser.parse_args(rest)
            return status(_profile_from(args))
        if command == "merge":
            from syco.orchestrate import merge
            parser = _profile_parser("Validate and merge model outputs")
            parser.add_argument("--allow-partial", action="store_true")
            args = parser.parse_args(rest)
            return merge(_profile_from(args), allow_partial=args.allow_partial)
        if command == "smoke":
            from syco.orchestrate import smoke
            parser = _profile_parser("Run the offline end-to-end smoke test")
            parser.add_argument("--model", default=None)
            args = parser.parse_args(rest)
            return smoke(_profile_from(args), args.model)
        if command == "doctor":
            from syco.orchestrate import doctor
            parser = _profile_parser("Validate the local environment")
            args = parser.parse_args(rest)
            return doctor(_profile_from(args))
        if command == "pipeline":
            return _pipeline(rest)
        print(f"unknown command: {command}\n", file=sys.stderr)
        print(HELP, file=sys.stderr)
        return 2
    except (FileNotFoundError, RuntimeError, ValueError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
