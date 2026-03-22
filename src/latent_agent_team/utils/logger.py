"""
utils/logger.py — Structured JSONL rollout recorder.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


@dataclass
class StepRecord:
    """One team step serialised for analysis / replay."""
    episode_id: str
    step_idx: int
    timestamp: str
    task_instruction: str
    observation: str            # truncated to 512 chars
    action: Dict[str, Any]
    reward: float
    done: bool
    routes: List[str]           # e.g. ["planner→browser", "verifier→planner"]
    k_values: Dict[str, int]    # edge → K tokens used
    total_k: int
    audit_texts: Dict[str, str] # edge → decoded latent text
    codebook_ids: Dict[str, Any]
    verification_label: str     # pass / fail / uncertain
    disagreement_score: float
    failure_tags: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeRecord:
    """Full episode summary appended at episode end."""
    episode_id: str
    task_instruction: str
    benchmark: str
    backbone: str
    comm_mode: str
    success: bool
    reward: float
    total_steps: int
    total_k_used: int
    mean_k: float
    num_routing_edges: int
    elapsed_sec: float
    failure_tags: List[str]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class RolloutLogger:
    """
    Thread-safe JSONL logger for rollout data.

    Each call to log_step() appends one StepRecord line.
    Each call to log_episode() appends one EpisodeRecord line.

    Files are rotated when they exceed max_mb megabytes.
    Set compress=True to write .jsonl.gz files.
    """

    def __init__(
        self,
        log_dir: str,
        run_name: str = "run",
        compress: bool = False,
        max_mb: float = 200.0,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self.compress = compress
        self.max_bytes = int(max_mb * 1024 * 1024)

        self._step_file: Optional[Any] = None
        self._ep_file: Optional[Any] = None
        self._lock = threading.Lock()
        self._step_path: Optional[Path] = None
        self._ep_path: Optional[Path] = None
        self._open_files()

    # ── File management ───────────────────────────────────────────────

    def _open_files(self) -> None:
        ext = ".jsonl.gz" if self.compress else ".jsonl"
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        self._step_path = self.log_dir / f"{self.run_name}_steps_{ts}{ext}"
        self._ep_path   = self.log_dir / f"{self.run_name}_episodes_{ts}{ext}"

        open_fn = gzip.open if self.compress else open
        mode = "wt" if self.compress else "w"
        self._step_file = open_fn(self._step_path, mode, encoding="utf-8")
        self._ep_file   = open_fn(self._ep_path,   mode, encoding="utf-8")
        log.debug("Opened log files: %s, %s", self._step_path, self._ep_path)

    def _rotate_if_needed(self) -> None:
        """Rotate step file if it exceeds max_bytes."""
        if self._step_path and self._step_path.exists():
            if self._step_path.stat().st_size >= self.max_bytes:
                self._step_file.close()
                self._ep_file.close()
                self._open_files()

    # ── Public API ────────────────────────────────────────────────────

    def log_step(self, record: StepRecord) -> None:
        with self._lock:
            self._rotate_if_needed()
            self._step_file.write(json.dumps(asdict(record), default=str) + "\n")
            self._step_file.flush()

    def log_episode(self, record: EpisodeRecord) -> None:
        with self._lock:
            self._ep_file.write(json.dumps(asdict(record), default=str) + "\n")
            self._ep_file.flush()

    def close(self) -> None:
        with self._lock:
            if self._step_file:
                self._step_file.close()
            if self._ep_file:
                self._ep_file.close()

    def __enter__(self) -> "RolloutLogger":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ── Convenience builders ──────────────────────────────────────────

    @staticmethod
    def make_step_record(
        episode_id: str,
        step_idx: int,
        task_instruction: str,
        observation: str,
        action: Dict,
        reward: float,
        done: bool,
        step_info: Dict,
    ) -> StepRecord:
        return StepRecord(
            episode_id=episode_id,
            step_idx=step_idx,
            timestamp=datetime.utcnow().isoformat(),
            task_instruction=task_instruction[:256],
            observation=observation[:512],
            action=action,
            reward=reward,
            done=done,
            routes=step_info.get("routes", []),
            k_values=step_info.get("k_values", {}),
            total_k=step_info.get("total_k", 0),
            audit_texts={
                k: v[:200] for k, v in step_info.get("audit_texts", {}).items()
            },
            codebook_ids=step_info.get("codebook_ids", {}),
            verification_label=step_info.get("verification", "uncertain"),
            disagreement_score=float(step_info.get("disagreement", 0.0)),
            failure_tags=step_info.get("failure_tags", []),
            meta=step_info.get("meta", {}),
        )
