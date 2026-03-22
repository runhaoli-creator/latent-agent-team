"""
mind2web_wrapper.py — Wrapper for Mind2Web benchmark environment.

Mind2Web provides 2,350 real-world web tasks across 137 websites, 31 domains.
  - Training data: public
  - Test data: encrypted (use official evaluation server)

Official repo: https://github.com/OSU-NLP-Group/Mind2Web
Data format: each task has a sequence of (observation, action) pairs
  - observation: raw HTML / DOM tree
  - action: {op: click|type|select, value: "...", element_id: "..."}

This wrapper handles:
  1. Loading and parsing Mind2Web JSONL data files
  2. Providing step-by-step task environments
  3. Evaluating with official metrics: element accuracy, operation F1, step/task success
  4. Optional live browser interaction via Playwright (for full DOM rendering)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class Mind2WebTask:
    task_id: str
    website: str
    domain: str
    subdomain: str
    confirmed_task: str
    annotation_id: str
    actions: List[Dict]   # list of {action_reprs, operation, raw_html, ...}
    split: str            # "train" | "test_task" | "test_website" | "test_domain"


@dataclass
class Mind2WebStep:
    step_idx: int
    raw_html: str
    cleaned_html: str
    operation: Dict  # {op: ..., value: ..., original_op: ...}
    action_reprs: List[str]
    pos_candidates: List[Dict]  # positive element candidates
    neg_candidates: List[Dict]  # negative element candidates


class Mind2WebEnv:
    """
    Mind2Web environment wrapper for sequential task execution.

    Usage:
        env = Mind2WebEnv(data_dir="data/mind2web", split="train")
        for task in env.iter_tasks():
            obs = env.reset(task)
            for step in range(max_steps):
                action = model_predict(obs)
                obs, reward, done, info = env.step(action)
                if done:
                    break
    """

    # Official Mind2Web data splits
    SPLITS = ["train", "test_task", "test_website", "test_domain"]

    def __init__(
        self,
        data_dir: str = "data/mind2web",
        split: str = "train",
        max_html_length: int = 4096,
        use_playwright: bool = False,
    ):
        self.data_dir = data_dir
        self.split = split
        self.max_html_length = max_html_length
        self.use_playwright = use_playwright
        self._current_task: Optional[Mind2WebTask] = None
        self._current_step_idx: int = 0
        self.tasks: List[Mind2WebTask] = []
        self._load_tasks()

    def _load_tasks(self) -> None:
        """Load tasks from Mind2Web data directory."""
        # Try multiple common file formats
        possible_files = [
            os.path.join(self.data_dir, f"{self.split}.json"),
            os.path.join(self.data_dir, f"data_{self.split}.json"),
            os.path.join(self.data_dir, self.split, "data.json"),
        ]

        for fpath in possible_files:
            if os.path.exists(fpath):
                with open(fpath, encoding="utf-8") as f:
                    raw_data = json.load(f)
                if isinstance(raw_data, list):
                    for d in raw_data:
                        task = self._parse_task(d, self.split)
                        if task:
                            self.tasks.append(task)
                logger.info(
                    f"Mind2Web: loaded {len(self.tasks)} tasks from {fpath} (split={self.split})"
                )
                return

        logger.warning(
            f"Mind2Web data not found in {self.data_dir} for split '{self.split}'. "
            "Please download from https://github.com/OSU-NLP-Group/Mind2Web"
        )

    def _parse_task(self, d: Dict, split: str) -> Optional[Mind2WebTask]:
        """Parse a raw task dict into Mind2WebTask."""
        try:
            actions = []
            for ann in d.get("annotations", []):
                for act in ann.get("actions", []):
                    actions.append(act)
            return Mind2WebTask(
                task_id=d.get("task_id", ""),
                website=d.get("website", ""),
                domain=d.get("domain", ""),
                subdomain=d.get("subdomain", ""),
                confirmed_task=d.get("confirmed_task", ""),
                annotation_id=d.get("annotation_id", ""),
                actions=actions,
                split=split,
            )
        except (KeyError, TypeError):
            return None

    def reset(self, task: Optional[Mind2WebTask] = None) -> Dict[str, Any]:
        """Reset environment for a new task. Returns initial observation."""
        if task is None and self.tasks:
            task = self.tasks[0]
        self._current_task = task
        self._current_step_idx = 0

        if task is None:
            return {"observation": "", "task": "", "dom": "", "step": 0}

        first_action = task.actions[0] if task.actions else {}
        return {
            "observation": first_action.get("raw_html", "")[:self.max_html_length],
            "task": task.confirmed_task,
            "dom": first_action.get("cleaned_html", ""),
            "step": 0,
            "website": task.website,
            "domain": task.domain,
        }

    def step(
        self,
        action: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Execute one step and evaluate against ground truth.

        Args:
            action: Predicted action dict {action: ..., element: ..., value: ...}

        Returns:
            obs: Next observation
            reward: 1.0 if step correct, 0.0 if wrong
            done: True if task complete
            info: Evaluation details
        """
        if self._current_task is None:
            return {"observation": ""}, 0.0, True, {}

        # Check if we're past end of task
        if self._current_step_idx >= len(self._current_task.actions):
            return {"observation": ""}, 0.0, True, {"reason": "task_complete"}

        # Get ground truth for current step
        gt = self._current_task.actions[self._current_step_idx]
        gt_op = gt.get("operation", {})
        gt_action_type = gt_op.get("op", "").lower()
        gt_element = gt_op.get("value", "")

        # Evaluate prediction
        pred_action_type = str(action.get("action", "")).lower()
        pred_element = str(action.get("element", action.get("value", "")))

        element_correct = pred_element.lower().strip() == gt_element.lower().strip()
        op_correct = pred_action_type == gt_action_type
        step_correct = element_correct and op_correct
        reward = 1.0 if step_correct else 0.0

        # Advance
        self._current_step_idx += 1
        done = self._current_step_idx >= len(self._current_task.actions)

        # Build next observation
        if not done:
            next_act = self._current_task.actions[self._current_step_idx]
            obs = {
                "observation": next_act.get("raw_html", "")[:self.max_html_length],
                "dom": next_act.get("cleaned_html", ""),
                "step": self._current_step_idx,
                "task": self._current_task.confirmed_task,
            }
        else:
            obs = {
                "observation": "",
                "step": self._current_step_idx,
                "task": self._current_task.confirmed_task,
            }

        info = {
            "element_correct": element_correct,
            "op_correct": op_correct,
            "step_correct": step_correct,
            "gt_action": gt_action_type,
            "gt_element": gt_element,
            "pred_action": pred_action_type,
            "pred_element": pred_element,
        }

        return obs, reward, done, info

    def iter_tasks(self, max_tasks: Optional[int] = None) -> List[Mind2WebTask]:
        """Iterate over all tasks in the split."""
        tasks = self.tasks[:max_tasks] if max_tasks else self.tasks
        return tasks

    def get_dom_candidates(self, step_idx: int) -> List[str]:
        """Get candidate DOM elements for element selection (Mind2Web eval)."""
        if self._current_task is None:
            return []
        if step_idx >= len(self._current_task.actions):
            return []
        act = self._current_task.actions[step_idx]
        pos = [c.get("backend_node_id", "") for c in act.get("pos_candidates", [])]
        neg = [c.get("backend_node_id", "") for c in act.get("neg_candidates", [])[:10]]
        return pos + neg

    @staticmethod
    def compute_metrics(
        step_results: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Compute official Mind2Web metrics from a list of step evaluation results.

        Returns:
            element_accuracy, operation_f1, step_success_rate, task_success_rate
        """
        if not step_results:
            return {"element_accuracy": 0, "operation_f1": 0, "step_success": 0, "task_success": 0}

        elem_correct = sum(r.get("element_correct", False) for r in step_results)
        elem_total = len(step_results)
        element_accuracy = elem_correct / elem_total

        op_tp = sum(1 for r in step_results if r.get("op_correct") and r.get("element_correct"))
        op_fp = sum(1 for r in step_results if not r.get("op_correct"))
        op_fn = sum(1 for r in step_results if r.get("op_correct") and not r.get("element_correct"))

        precision = op_tp / max(1, op_tp + op_fp)
        recall = op_tp / max(1, op_tp + op_fn)
        if precision + recall > 0:
            operation_f1 = 2 * precision * recall / (precision + recall)
        else:
            operation_f1 = 0.0

        step_success = sum(r.get("step_correct", False) for r in step_results) / elem_total

        # Task success: all steps correct within a task (approximate)
        # In reality, this requires episode-level grouping
        task_success = step_success  # placeholder for now

        return {
            "element_accuracy": element_accuracy,
            "operation_f1": operation_f1,
            "step_success": step_success,
            "task_success": task_success,
        }
