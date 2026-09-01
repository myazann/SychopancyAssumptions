"""Where this study keeps its data and configuration. Stdlib only.

Every module resolves paths through here rather than off its own `__file__`,
so moving a module can never silently repoint it at a different directory.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CONFIG_DIR = ROOT / "config"
MODELS_PATH = CONFIG_DIR / "models.yaml"

# The base data (personas, prompts, and any already-collected answers). Not in
# git -- see README for the Drive link. Override with SYCO_DATA_DIR.
DATA_DIR = Path(os.environ.get("SYCO_DATA_DIR", ROOT / "files"))

PERSONA_PATH = DATA_DIR / "base_data_persona.gz"
PROMPT_PATH = DATA_DIR / "base_data_prompt.gz"
DEMOGRAPHICS_PATH = DATA_DIR / "personas_demographics_vulnerability_final.csv"

# Where runs land.
RESULTS_DIR = Path(os.environ.get("SYCO_RESULTS_DIR", ROOT / "results"))
