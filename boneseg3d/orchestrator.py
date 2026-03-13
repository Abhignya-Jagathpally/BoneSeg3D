"""
boneseg3d/orchestrator.py
==========================
Manages git worktrees and dispatches Claude Code subagents.

Two dispatch modes
──────────────────
  INLINE   — calls the task function directly in the current process
             (used when the worktree already exists or for unit tests)

  SUBAGENT — forks a `claude` CLI subprocess inside the worktree
             (used for heavy GPU tasks that benefit from isolation)

The Orchestrator decides which mode to use based on cfg.orchestration.mode.

Worktree lifecycle
──────────────────
  1. git worktree add <path> -b <branch>   (idempotent — skips if exists)
  2. task_fn(cfg) runs inside that path
  3. On success → write CHECKPOINT.json into worktree
  4. git worktree remove (optional, controlled by cfg.orchestration.keep_worktrees)
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Coroutine

from boneseg3d.utils.checkpoint  import mark_complete
from boneseg3d.utils.logging_utils import get_logger

log = get_logger("orchestrator")


class WorktreeError(RuntimeError):
    """Raised when git worktree operations fail."""


class Orchestrator:
    """
    Coordinates isolated worktree branches and Claude Code subagent dispatch.

    Parameters
    ----------
    repo_root : Path
        Absolute path to the git repository root (where .git lives).
    cfg : SimpleNamespace-like
        Project config loaded from default.yaml.
    """

    def __init__(self, repo_root: Path, cfg: Any) -> None:
        self.repo_root   = repo_root.resolve()
        self.cfg         = cfg
        self.worktrees_root = self.repo_root / ".worktrees"
        self.worktrees_root.mkdir(parents=True, exist_ok=True)
        self._active_worktrees: list[Path] = []

    # ──────────────────────────────────────────────────────────────────────────
    #  Public interface
    # ──────────────────────────────────────────────────────────────────────────

    async def run_in_worktree(
        self,
        branch: str,
        task_fn: Callable,
        task_args: tuple,
        checkpoint_key: str,
    ) -> None:
        """
        Run *task_fn* inside an isolated git worktree on *branch*.

        If cfg.orchestration.mode == "subagent", generates a Claude Code
        prompt and forks a `claude` CLI subprocess. Otherwise calls task_fn
        directly (useful for development / CI).
        """
        wt_path = self._provision_worktree(branch)
        mode    = getattr(self.cfg.orchestration, "mode", "inline")

        try:
            if mode == "subagent":
                await self._dispatch_claude_subagent(
                    worktree_path=wt_path,
                    branch=branch,
                    task_fn=task_fn,
                    checkpoint_key=checkpoint_key,
                )
            else:
                log.info(f"[INLINE] Running {task_fn.__name__} in {wt_path}")
                # Run blocking task in a thread so asyncio stays responsive
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, task_fn, *task_args)
                mark_complete(checkpoint_key)
        finally:
            if not getattr(self.cfg.orchestration, "keep_worktrees", True):
                self._teardown_worktree(wt_path, branch)

    # ──────────────────────────────────────────────────────────────────────────
    #  Worktree management
    # ──────────────────────────────────────────────────────────────────────────

    def _provision_worktree(self, branch: str) -> Path:
        """
        Create (or reuse) a git worktree at .worktrees/<sanitised-branch>.
        Returns the absolute path to the worktree directory.
        """
        safe_name = branch.replace("/", "-").replace(" ", "_")
        wt_path   = self.worktrees_root / safe_name

        if wt_path.exists():
            log.info(f"[WORKTREE] Reusing existing worktree: {wt_path}")
            return wt_path

        log.info(f"[WORKTREE] Creating worktree {wt_path} on branch '{branch}'")
        result = subprocess.run(
            ["git", "worktree", "add", str(wt_path), "-b", branch],
            cwd=str(self.repo_root),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            # Branch already exists remotely — just check it out
            result2 = subprocess.run(
                ["git", "worktree", "add", str(wt_path), branch],
                cwd=str(self.repo_root),
                capture_output=True,
                text=True,
            )
            if result2.returncode != 0:
                raise WorktreeError(
                    f"Could not create worktree for branch '{branch}':\n"
                    f"{result.stderr}\n{result2.stderr}"
                )

        self._active_worktrees.append(wt_path)
        log.info(f"[WORKTREE] ✓ Provisioned: {wt_path}")
        return wt_path

    def _teardown_worktree(self, wt_path: Path, branch: str) -> None:
        """Remove a worktree and optionally delete its branch."""
        if not wt_path.exists():
            return
        log.info(f"[WORKTREE] Removing {wt_path}")
        subprocess.run(
            ["git", "worktree", "remove", str(wt_path), "--force"],
            cwd=str(self.repo_root),
            check=False,
        )
        if wt_path in self._active_worktrees:
            self._active_worktrees.remove(wt_path)

    def list_active_worktrees(self) -> list[dict]:
        """Return structured info about all active git worktrees."""
        result = subprocess.run(
            ["git", "worktree", "list", "--porcelain"],
            cwd=str(self.repo_root),
            capture_output=True,
            text=True,
        )
        trees, current = [], {}
        for line in result.stdout.splitlines():
            if line.startswith("worktree "):
                if current:
                    trees.append(current)
                current = {"path": line.split(" ", 1)[1]}
            elif line.startswith("branch "):
                current["branch"] = line.split(" ", 1)[1]
            elif line.startswith("HEAD "):
                current["head"] = line.split(" ", 1)[1]
        if current:
            trees.append(current)
        return trees

    # ──────────────────────────────────────────────────────────────────────────
    #  Claude Code subagent dispatch
    # ──────────────────────────────────────────────────────────────────────────

    async def _dispatch_claude_subagent(
        self,
        worktree_path: Path,
        branch: str,
        task_fn: Callable,
        checkpoint_key: str,
    ) -> None:
        """
        Fork a `claude` CLI subprocess inside the worktree.

        The prompt is built from the task function's docstring + a standard
        BoneSeg3D context header, so each subagent understands the project
        without re-reading CLAUDE.md from scratch.
        """
        prompt = self._build_subagent_prompt(
            task_fn=task_fn,
            checkpoint_key=checkpoint_key,
            worktree_path=worktree_path,
        )

        prompt_file = worktree_path / f".claude_prompt_{checkpoint_key}.txt"
        prompt_file.write_text(prompt, encoding="utf-8")

        log.info(f"[SUBAGENT] Launching claude for '{checkpoint_key}' in {worktree_path}")

        proc = await asyncio.create_subprocess_exec(
            "claude",
            "--print",               # non-interactive, print response and exit
            "--dangerously-skip-permissions",  # allow file writes inside worktree
            f"@{prompt_file}",       # @ prefix reads prompt from file
            cwd=str(worktree_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, "BONESEG3D_CHECKPOINT_KEY": checkpoint_key},
        )

        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            log.error(
                f"[SUBAGENT] '{checkpoint_key}' exited with code {proc.returncode}\n"
                f"STDERR:\n{stderr.decode()}"
            )
            raise RuntimeError(
                f"Claude subagent failed for task '{checkpoint_key}' "
                f"(exit code {proc.returncode})"
            )

        log.info(f"[SUBAGENT] ✓ '{checkpoint_key}' complete.")
        self._write_worktree_checkpoint(worktree_path, checkpoint_key)
        mark_complete(checkpoint_key)

    def _build_subagent_prompt(
        self,
        task_fn: Callable,
        checkpoint_key: str,
        worktree_path: Path,
    ) -> str:
        """
        Compose the full prompt sent to the claude CLI subprocess.
        Includes: project context header + task docstring + completion contract.
        """
        task_doc = textwrap.dedent(task_fn.__doc__ or f"Implement {task_fn.__name__}").strip()

        return textwrap.dedent(f"""
            # BoneSeg3D Subagent Task: {checkpoint_key}
            # Worktree: {worktree_path}
            # Branch:   derived from checkpoint key

            ## Project Context
            You are a subagent for BoneSeg3D, an end-to-end 3D bone lesion
            segmentation pipeline for Multiple Myeloma CT scans.
            Stack: MONAI · nnU-Net · PyTorch · SimpleElastix · Streamlit
            Compute: UNT H100 GPU
            Dataset: TCIA Multiple Myeloma CT collection

            ## Your Task
            {task_doc}

            ## Completion Contract
            When done:
            1. Write all output files to {worktree_path}
            2. Run any required tests or sanity checks
            3. Print a summary of what was implemented
            4. Write a CHECKPOINT.json to {worktree_path}/checkpoints/{checkpoint_key}.json
               with keys: step, completed_at (ISO8601), files_written (list), metrics (dict)

            Do NOT modify files outside {worktree_path}.
            Do NOT commit or push. The orchestrator handles git operations.
        """).strip()

    @staticmethod
    def _write_worktree_checkpoint(worktree_path: Path, key: str) -> None:
        """Write a local CHECKPOINT.json inside the worktree for traceability."""
        ckpt_dir = worktree_path / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "step":         key,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "worktree":     str(worktree_path),
        }
        (ckpt_dir / f"{key}.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )