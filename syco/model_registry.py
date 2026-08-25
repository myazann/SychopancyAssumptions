"""Model registry: aliases, provider routing, and quantization handling.

Adapted from `core/model_registry.py` in myazan/LLM-Self-Concept, trimmed to
what this study needs. The parts that carry over unchanged are the ones worth
keeping identical across projects, so a model entry can be copied between the
two repos verbatim:

  * the alias table lives in `config/models.yaml`, so adding a model is a config
    edit rather than a code change;
  * backend routing is decided by the *shape* of the ref, with an explicit
    `backend:` override --  `*.gguf` / `*-GGUF` -> llamacpp, `org/name` -> hf,
    a bare name -> the family's API;
  * GGUF filenames are resolved from the repo listing at load time (quant
    filenames drift: Q4_K_M / UD-Q4_K_M / sharded -00001-of-0000N), so YAML pins
    the quant TAG and the loader finds the file.

Dropped relative to the source: the release-window/anchor bookkeeping and the
quantization sensitivity sweep, neither of which this study administers.
Retained: `release_date`, `family`, `generation` and `quantization`, because
they are what a results row needs to be reproducible.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path
from typing import Optional

import yaml

from syco.paths import MODELS_PATH

CONFIG_PATH = MODELS_PATH

LLAMACPP_BACKEND = "llamacpp"
HUGGINGFACE_BACKEND = "hf"
OPENAI_BACKEND = "openai"
ANTHROPIC_BACKEND = "anthropic"
MOCK_BACKEND = "mock"

API_BACKENDS = frozenset({OPENAI_BACKEND, ANTHROPIC_BACKEND})
LOCAL_BACKENDS = frozenset({LLAMACPP_BACKEND, HUGGINGFACE_BACKEND})

_FAMILY_API_BACKEND = {"claude": ANTHROPIC_BACKEND, "gpt": OPENAI_BACKEND}


@dataclass(frozen=True)
class Reasoning:
    """Whether the raw model reasons unprompted, and whether we can force it.

    The probe wants a verbalized assumption block, not a hidden reasoning trace:
    a model that thinks privately and then prints three tidy mental models has
    not shown us the assumptions that drove the answer. So runs default to
    thinking OFF wherever the backend can assert it, and every row records what
    was actually asked for.
    """
    thinks_by_default: bool = False
    control: str = "none"        # thinking_param | effort | template_toggle | none
    controllable: bool = True

    @property
    def label(self) -> str:
        return f"{self.control}{'' if self.controllable else '(fixed)'}"


@dataclass(frozen=True)
class Quantization:
    """How the weights are quantized. `format` picks the loader path."""
    format: str = "none"            # gguf | hf | none
    quant: Optional[str] = None     # gguf tag, e.g. Q4_K_M / Q8_0
    method: Optional[str] = None    # hf method, e.g. bnb-4bit / gptq-int4 / fp8
    resolved_file: Optional[str] = None   # filled in by resolve_gguf_file()

    @property
    def label(self) -> str:
        if self.format == "gguf":
            return f"gguf:{self.quant}"
        if self.format == "hf":
            return f"hf:{self.method or 'none'}"
        return "none"


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    family: str
    ref: str
    backend: str
    generation: str = ""
    release_date: Optional[date] = None
    kind: str = "instruct"          # instruct | base
    quantization: Quantization = field(default_factory=Quantization)
    hf_id: Optional[str] = None     # unquantized upstream, for provenance
    tokenizer_id: Optional[str] = None  # public/template source; defaults to hf_id
    params_total_b: Optional[float] = None
    size_tier: Optional[str] = None
    enabled: bool = True
    reasoning: Reasoning = field(default_factory=Reasoning)
    # -- decoding. Defaults follow the paper's get_assumptions.py: temperature
    # 0.7, top_p 0.9, and a large output budget, because one cell has to fit a
    # k-model JSON block AND a full advice reply.
    temperature: float = 0.7
    top_p: float = 0.9
    max_output_tokens: int = 2048
    # -- throughput
    batch_size: int = 8             # local backends: prompts per generate() call
    max_workers: int = 4            # API backends: concurrent requests
    runtime: dict = field(default_factory=dict)
    notes: str = ""

    @property
    def is_local(self) -> bool:
        return self.backend in LOCAL_BACKENDS

    @property
    def is_api(self) -> bool:
        return self.backend in API_BACKENDS

    def safe_dir_name(self) -> str:
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", self.alias)
        return safe.strip("._-") or "model"

    def provenance(self) -> dict:
        """Everything a results row needs to be reproducible."""
        return {
            "model_id": self.alias,
            "model_ref": self.ref,
            "model_family": self.family,
            "model_generation": self.generation,
            "model_release_date": self.release_date.isoformat() if self.release_date else None,
            "quantization": self.quantization.label,
            "quantized_file": self.quantization.resolved_file,
            "backend": self.backend,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_output_tokens": self.max_output_tokens,
        }


# ---------------------------------------------------------------------------
# routing
# ---------------------------------------------------------------------------
def infer_backend(ref: str, family: str = "") -> str:
    """*.gguf / *-GGUF -> llamacpp ; "org/name" -> hf ; bare name -> family API."""
    normalized = ref.strip()
    lowered = normalized.lower()
    if lowered.endswith(".gguf") or lowered.endswith("-gguf"):
        return LLAMACPP_BACKEND
    if "/" in normalized:
        return HUGGINGFACE_BACKEND
    return _FAMILY_API_BACKEND.get(family.lower(), OPENAI_BACKEND)


# ---------------------------------------------------------------------------
# GGUF file resolution
# ---------------------------------------------------------------------------
def split_hf_gguf_ref(ref: str) -> tuple:
    """'org/repo/file.gguf' -> ('org/repo', 'file.gguf'); 'org/repo' -> (ref, None)."""
    parts = [p for p in ref.strip().split("/") if p]
    if len(parts) < 2:
        raise ValueError(f"GGUF ref must be at least 'org/repo': {ref!r}")
    if len(parts) == 2:
        return "/".join(parts), None
    return "/".join(parts[:2]), "/".join(parts[2:])


# Auxiliary .gguf files that are not the model: vision projectors, multi-token
# prediction heads, speculative drafts.
_AUX_GGUF = re.compile(r"(^|/)(mmproj|mtp-|draft-)|(^|/)(MTP|mmproj)/", re.IGNORECASE)


def _cached_gguf_files(repo_id: str) -> list:
    """`.gguf` filenames already in the local HF cache -- offline fallback, so a
    flaky connection cannot kill a run whose weights are already on disk."""
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except ImportError:
        return []
    snaps = Path(HF_HUB_CACHE) / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    if not snaps.is_dir():
        return []
    seen = {}
    for snap in sorted(snaps.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        for f in snap.rglob("*.gguf"):
            seen.setdefault(str(f.relative_to(snap)), True)
    return list(seen)


def resolve_gguf_file(repo_id: str, quant: str, prefer_dynamic: bool = False) -> str:
    """Find the .gguf in `repo_id` matching the quant tag.

    Raises with the available quants listed, which beats a 404 on a hardcoded
    filename. For sharded quants it returns the first shard; llama.cpp loads the
    rest.
    """
    try:
        from huggingface_hub import list_repo_files
    except ImportError as err:
        raise RuntimeError(
            "Resolving GGUF filenames needs `huggingface_hub`, or pin the exact "
            "file in models.yaml via `gguf_file:`."
        ) from err

    token = os.environ.get("HF_TOKEN")
    try:
        listing = list_repo_files(repo_id, token=token)
    except Exception as err:
        listing = _cached_gguf_files(repo_id)
        if not listing:
            raise RuntimeError(
                f"Could not list {repo_id} ({type(err).__name__}: {err}) and no "
                ".gguf is cached locally. Connect for the first download, or pin "
                "the file in models.yaml via `gguf_file:`."
            ) from err
    files = [f for f in listing if f.lower().endswith(".gguf") and not _AUX_GGUF.search(f)]
    if not files:
        raise RuntimeError(f"No model .gguf files in {repo_id}.")

    def shard_index(name: str) -> int:
        m = re.search(r"-(\d{5})-of-\d{5}\.gguf$", name)
        return int(m.group(1)) if m else 0

    tags = [f"UD-{quant}", quant] if prefer_dynamic else [quant, f"UD-{quant}"]
    for tag in tags:
        # Anchor the tag between separators so Q4_K_M doesn't match Q4_K_M_XL.
        pattern = re.compile(rf"(^|[-_./]){re.escape(tag)}(\.gguf$|-\d{{5}}-of-)", re.IGNORECASE)
        matches = sorted((f for f in files if pattern.search(f)), key=shard_index)
        if matches:
            return matches[0]

    available = sorted({_quant_tag_of(f) for f in files})
    raise RuntimeError(f"No {quant} file in {repo_id}. Available quants: {', '.join(available)}")


def _quant_tag_of(filename: str) -> str:
    m = re.search(r"((?:UD-)?(?:IQ|Q)\d[^./-]*(?:_[A-Z0-9]+)*|BF16|F16|F32|MXFP4)",
                  filename, re.IGNORECASE)
    return m.group(1) if m else filename


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
class ModelRegistry:
    def __init__(self, specs: list, meta: dict):
        self._by_alias = {s.alias: s for s in specs}
        if len(self._by_alias) != len(specs):
            dupes = [a for a in {s.alias for s in specs}
                     if sum(x.alias == a for x in specs) > 1]
            raise ValueError(f"Duplicate aliases in models.yaml: {dupes}")
        self.meta = meta

    def __len__(self) -> int:
        return len(self._by_alias)

    def __iter__(self):
        return iter(self._by_alias.values())

    def get(self, alias: str) -> ModelSpec:
        key = alias.strip()
        if key in self._by_alias:
            return self._by_alias[key]
        for spec in self._by_alias.values():     # accept a resolved ref too
            if key in (spec.ref, spec.hf_id):
                return spec
        raise KeyError(f"Unknown model {alias!r}. Known: {', '.join(sorted(self._by_alias))}")

    def aliases(self) -> tuple:
        return tuple(self._by_alias)

    def select(self, families=None, backends=None, include_disabled: bool = False) -> list:
        out = []
        for spec in self._by_alias.values():
            if not include_disabled and not spec.enabled:
                continue
            if families and spec.family not in families:
                continue
            if backends and spec.backend not in backends:
                continue
            out.append(spec)
        return sorted(out, key=lambda s: (s.family, s.alias))

    def with_resolved_quant(self, spec: ModelSpec) -> ModelSpec:
        """`spec` with the concrete GGUF filename filled in. Network call --
        done once per model at load time, never per cell."""
        q = spec.quantization
        if q.format != "gguf" or q.resolved_file:
            return spec
        repo_id, pinned = split_hf_gguf_ref(spec.ref)
        prefer = _deep_get(self.meta, "defaults", "quantization", "gguf",
                           "prefer_unsloth_dynamic", default=False)
        filename = pinned or resolve_gguf_file(repo_id, q.quant, prefer_dynamic=bool(prefer))
        return replace(spec, quantization=replace(q, resolved_file=filename))


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def _deep_get(d: dict, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _build_quantization(entry: dict, defaults: dict) -> Quantization:
    raw = entry.get("quantization")
    if raw is None:
        return Quantization()
    fmt = raw.get("format", "none")
    if fmt == "gguf":
        return Quantization(
            format="gguf",
            quant=raw.get("quant") or _deep_get(defaults, "quantization", "gguf", "quant"),
            resolved_file=entry.get("gguf_file"),
        )
    if fmt == "hf":
        return Quantization(
            format="hf",
            method=raw.get("method") or _deep_get(defaults, "quantization", "hf", "method"),
        )
    return Quantization()


def _build_reasoning(entry: dict, by_family: dict) -> Reasoning:
    """Per-model override > family default > none."""
    profile = dict(by_family.get(entry.get("family", ""), {}))
    profile.update(entry.get("reasoning") or {})
    if not profile:
        return Reasoning()
    return Reasoning(
        thinks_by_default=bool(profile.get("thinks_by_default", False)),
        control=profile.get("control", "none"),
        controllable=bool(profile.get("controllable", True)),
    )


def load_registry(path=CONFIG_PATH) -> ModelRegistry:
    with open(path, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}

    defaults = doc.get("defaults", {}) or {}
    runtime_defaults = defaults.get("runtime", {}) or {}
    reasoning_by_family = doc.get("reasoning_by_family", {}) or {}
    specs = []

    for entry in doc.get("models", []):
        family = entry.get("family", "")
        ref = entry["ref"]
        backend = entry.get("backend") or infer_backend(ref, family)
        runtime = dict(runtime_defaults.get(backend, {}) or {})
        runtime.update(entry.get("runtime", {}) or {})

        released = entry.get("release_date")
        if isinstance(released, str):
            released = date.fromisoformat(released)

        specs.append(ModelSpec(
            alias=entry["alias"],
            family=family,
            ref=ref,
            backend=backend,
            generation=entry.get("generation", ""),
            release_date=released,
            kind=entry.get("kind", "instruct"),
            quantization=_build_quantization(entry, defaults),
            hf_id=entry.get("hf_id"),
            tokenizer_id=entry.get("tokenizer_id"),
            params_total_b=entry.get("params_total_b"),
            size_tier=entry.get("size_tier"),
            enabled=bool(entry.get("enabled", True)),
            reasoning=_build_reasoning(entry, reasoning_by_family),
            temperature=entry.get("temperature", defaults.get("temperature", 0.7)),
            top_p=entry.get("top_p", defaults.get("top_p", 0.9)),
            max_output_tokens=entry.get("max_output_tokens",
                                        defaults.get("max_output_tokens", 2048)),
            batch_size=entry.get("batch_size", defaults.get("batch_size", 8)),
            max_workers=entry.get("max_workers", defaults.get("max_workers", 4)),
            runtime=runtime,
            notes=(entry.get("notes") or "").strip(),
        ))

    return ModelRegistry(specs, doc)


if __name__ == "__main__":
    reg = load_registry()
    header = (f"{'alias':<22} {'family':<8} {'backend':<10} {'quant':<14} "
              f"{'temp':<5} {'maxtok':<7} {'ref'}")
    print(header)
    print("-" * len(header))
    for s in reg.select(include_disabled=True):
        flag = "" if s.enabled else "  (disabled)"
        print(f"{s.alias:<22} {s.family:<8} {s.backend:<10} {s.quantization.label:<14} "
              f"{s.temperature:<5} {s.max_output_tokens:<7} {s.ref}{flag}")
