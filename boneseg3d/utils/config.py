"""
boneseg3d/utils/config.py
===========================
YAML config loader that returns a SimpleNamespace tree so callers can
use dot-access (cfg.data.nifti_dir) without importing pydantic.

Usage
─────
  cfg = load_config("configs/default.yaml")
  print(cfg.monai.max_epochs)   # 300
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml


def _to_namespace(obj: Any) -> Any:
    """Recursively convert dicts to SimpleNamespace for dot-access."""
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(i) for i in obj]
    return obj


def load_config(path: str | Path) -> SimpleNamespace:
    """
    Load a YAML config file and return it as a nested SimpleNamespace.
    Environment variables of the form ${VAR} in values are substituted.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw = path.read_text(encoding="utf-8")

    # Simple env-var substitution: ${VAR} → os.environ.get(VAR, "")
    import re
    def _subst(match):
        return os.environ.get(match.group(1), "")
    raw = re.sub(r"\$\{(\w+)\}", _subst, raw)

    data = yaml.safe_load(raw)
    return _to_namespace(data)