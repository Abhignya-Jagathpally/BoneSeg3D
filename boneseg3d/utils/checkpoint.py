"""
boneseg3d/utils/checkpoint.py
================================
Checkpoint system — persists task completion state to JSON files so the
orchestrator can resume the pipeline after interruption without re-running
GPU-intensive steps.

Design principles
─────────────────
  • Idempotent:  calling mark_complete() twice for the same key is safe
  • Atomic:      write to a .tmp file then rename — avoids partial writes
  • Queryable:   load_all() returns the full checkpoint registry as a dict
  • Portable:    plain JSON — no database dependency
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CHECKPOINT_DIR = Path(os.environ.get("BONESEG3D_CHECKPOINT_DIR", "checkpoints"))


def _checkpoint_path(key: str) -> Path:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    return CHECKPOINT_DIR / f"{key}.json"


def mark_complete(key: str, metadata: dict[str, Any] | None = None) -> None:
    """
    Record that *key* has completed successfully.
    Metadata is arbitrary — callers store metrics, file paths, durations, etc.
    """
    record = {
        "key":          key,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "metadata":     metadata or {},
    }
    path    = _checkpoint_path(key)
    tmp     = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=2), encoding="utf-8")
    tmp.replace(path)   # atomic rename
    print(f"[CHECKPOINT] ✓ {key}  →  {path}")


def is_complete(key: str) -> bool:
    """Return True if *key* has a valid checkpoint file."""
    return _checkpoint_path(key).exists()


def load_checkpoint(key: str) -> dict | None:
    """Load and return the checkpoint record for *key*, or None if missing."""
    path = _checkpoint_path(key)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def clear_checkpoint(key: str) -> None:
    """Delete a checkpoint (forces the task to re-run on next pipeline invocation)."""
    path = _checkpoint_path(key)
    if path.exists():
        path.unlink()
        print(f"[CHECKPOINT] Cleared: {key}")


def load_all() -> dict[str, dict]:
    """Return a mapping of all checkpoint keys to their records."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    return {
        p.stem: json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(CHECKPOINT_DIR.glob("*.json"))
    }


def print_status() -> None:
    """Print a human-readable summary of all checkpoints."""
    all_ckpts = load_all()
    if not all_ckpts:
        print("[CHECKPOINT] No checkpoints found.")
        return
    print(f"\n{'Key':<50} {'Completed At':<30} {'Metadata Keys'}")
    print("-" * 100)
    for key, record in all_ckpts.items():
        meta_keys = ", ".join(record.get("metadata", {}).keys()) or "—"
        print(f"{key:<50} {record['completed_at']:<30} {meta_keys}")