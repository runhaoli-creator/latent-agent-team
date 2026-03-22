"""
agentbench_wrapper.py — Wrapper for AgentBench benchmark environments.

AgentBench covers 8 interactive environments:
  - OS: operating system shell commands
  - DB: SQL database queries
  - Knowledge Graph: SPARQL / graph traversal
  - Digital Card Game
  - Lateral Thinking
  - House Holding (embodied)
  - Web Shopping (subset)
  - Web Browsing

Official repo: https://github.com/THUDM/AgentBench
Protocol: GPT-4 as the primary evaluator; scores based on task completion + correctness

This wrapper provides:
  1. Stub implementations for each env type for offline testing
  2. Task loading from AgentBench format
  3. Reward computation based on expected outputs
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class AgentBenchTask:
    task_id: str
    env_type: str     # "os" | "db" | "kg" | "web" | "card" | "lateral" | "house"
    instruction: str
    expected_output: str
    init_commands: List[str] = field(default_factory=list)
    max_steps: int = 15
    metadata: Dict[str, Any] = field(default_factory=dict)


class OSEnvStub:
    """
    Lightweight stub for OS (shell) environment.
    Runs commands in a sandboxed subprocess.
    """

    SAFE_COMMANDS = {
        "ls", "pwd", "cat", "echo", "grep", "find", "wc", "head", "tail",
        "sort", "uniq", "cut", "awk", "sed", "diff", "stat",
    }

    def __init__(self, task: Optional[AgentBenchTask] = None):
        self.task = task
        self._cwd = "/tmp"
        self._history: List[str] = []

    def reset(self) -> str:
        self._history = []
        if self.task and self.task.init_commands:
            for cmd in self.task.init_commands[:3]:
                self._execute(cmd)
        instruction = self.task.instruction if self.task else "Complete the task."
        return f"$ pwd\n{self._cwd}\n\nTask: {instruction}"

    def _execute(self, command: str) -> str:
        """Execute a command safely."""
        # Only allow whitelisted commands for safety
        cmd_name = command.strip().split()[0] if command.strip() else ""
        if cmd_name not in self.SAFE_COMMANDS:
            return f"Command '{cmd_name}' not permitted in stub env."

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=self._cwd,
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.stdout[:500] + (result.stderr[:200] if result.stderr else "")
        except subprocess.TimeoutExpired:
            return "Command timed out."
        except Exception as e:
            return f"Error: {e}"

    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        self._history.append(action)
        output = self._execute(action)

        # Check if expected output is in result
        success = False
        if self.task and self.task.expected_output:
            success = self.task.expected_output.strip() in output

        reward = 1.0 if success else 0.0
        done = success or len(self._history) >= (self.task.max_steps if self.task else 10)

        obs = f"$ {action}\n{output}\n\n$ "
        return obs, reward, done, {"output": output, "success": success}


class DBEnvStub:
    """
    Lightweight stub for DB (SQL) environment.
    Uses Python's built-in sqlite3 for safe execution.
    """

    def __init__(self, task: Optional[AgentBenchTask] = None):
        self.task = task
        self._db_path = ":memory:"
        self._history: List[str] = []
        self._setup_demo_db()

    def _setup_demo_db(self) -> None:
        import sqlite3
        try:
            self._conn = sqlite3.connect(self._db_path)
            # Create demo tables
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS employees (
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    department TEXT,
                    salary REAL
                )
            """)
            self._conn.execute(
                "INSERT OR IGNORE INTO employees VALUES (1,'Alice','Engineering',95000)"
            )
            self._conn.execute(
                "INSERT OR IGNORE INTO employees VALUES (2,'Bob','Marketing',75000)"
            )
            self._conn.execute(
                "INSERT OR IGNORE INTO employees VALUES (3,'Carol','Engineering',105000)"
            )
            self._conn.commit()
        except Exception as e:
            logger.warning(f"DB setup error: {e}")
            self._conn = None

    def reset(self) -> str:
        self._history = []
        instruction = self.task.instruction if self.task else "Run SQL queries to complete the task."
        return (
            f"Database schema:\n"
            f"  employees(id, name, department, salary)\n\n"
            f"Task: {instruction}\n\n> "
        )

    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        self._history.append(action)
        result = ""

        # Extract SQL from action
        sql_match = re.search(r"(SELECT|INSERT|UPDATE|DELETE|CREATE).*", action, re.IGNORECASE | re.DOTALL)
        if sql_match:
            sql = sql_match.group(0)[:500]
            try:
                if self._conn:
                    cursor = self._conn.execute(sql)
                    rows = cursor.fetchall()
                    result = str(rows[:10])
            except Exception as e:
                result = f"SQL Error: {e}"
        else:
            result = "No SQL query found."

        success = bool(
            self.task and self.task.expected_output
            and self.task.expected_output.lower() in result.lower()
        )
        reward = 1.0 if success else 0.0
        done = success or len(self._history) >= (self.task.max_steps if self.task else 10)

        obs = f"> {action}\n{result}\n\n> "
        return obs, reward, done, {"result": result, "success": success}


class AgentBenchEnv:
    """
    AgentBench multi-environment wrapper.
    Supports: OS, DB, KG, Web, and other environments.
    """

    ENV_TYPES = ["os", "db", "kg", "web", "card", "lateral"]

    def __init__(
        self,
        task_dir: str = "data/agentbench",
        env_type: str = "os",
        max_episode_steps: int = 15,
    ):
        self.task_dir = task_dir
        self.env_type = env_type
        self.max_episode_steps = max_episode_steps
        self._current_env: Optional[Any] = None
        self._step_count = 0
        self.tasks = self._load_tasks()

    def _load_tasks(self) -> List[AgentBenchTask]:
        tasks = []

        # Try loading from standard AgentBench format
        possible_files = [
            os.path.join(self.task_dir, f"{self.env_type}_tasks.jsonl"),
            os.path.join(self.task_dir, self.env_type, "tasks.jsonl"),
            os.path.join(self.task_dir, "tasks.jsonl"),
        ]

        for fpath in possible_files:
            if os.path.exists(fpath):
                with open(fpath) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            d = json.loads(line)
                            tasks.append(AgentBenchTask(
                                task_id=d.get("id", d.get("task_id", "")),
                                env_type=self.env_type,
                                instruction=d.get("description", d.get("instruction", "")),
                                expected_output=d.get("answer", d.get("expected", "")),
                                init_commands=d.get("init_commands", []),
                                max_steps=d.get("max_steps", self.max_episode_steps),
                                metadata=d.get("metadata", {}),
                            ))
                        except (json.JSONDecodeError, KeyError) as e:
                            logger.warning(f"Skipping malformed task: {e}")
                logger.info(
                    f"AgentBench: loaded {len(tasks)} {self.env_type} tasks from {fpath}"
                )
                return tasks

        # Fall back to demo tasks
        logger.warning(
            f"AgentBench tasks not found in {self.task_dir}/{self.env_type}. "
            "Using demo tasks. Please install AgentBench from "
            "https://github.com/THUDM/AgentBench"
        )
        return self._make_demo_tasks()

    def _make_demo_tasks(self) -> List[AgentBenchTask]:
        if self.env_type == "os":
            return [
                AgentBenchTask(
                    task_id="os_demo_001",
                    env_type="os",
                    instruction="List all .txt files in the current directory",
                    expected_output="",
                    max_steps=5,
                ),
            ]
        elif self.env_type == "db":
            return [
                AgentBenchTask(
                    task_id="db_demo_001",
                    env_type="db",
                    instruction="Find all employees in the Engineering department",
                    expected_output="Engineering",
                    max_steps=5,
                ),
            ]
        return []

    def reset(self, task: Optional[AgentBenchTask] = None) -> str:
        """Reset environment for a task."""
        self._step_count = 0
        if task is None and self.tasks:
            task = self.tasks[0]

        if self.env_type == "os":
            self._current_env = OSEnvStub(task=task)
        elif self.env_type == "db":
            self._current_env = DBEnvStub(task=task)
        else:
            # Generic stub
            self._current_env = OSEnvStub(task=task)

        return self._current_env.reset()

    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        self._step_count += 1
        timeout = self._step_count >= self.max_episode_steps

        if self._current_env is None:
            return "Error: env not initialized", 0.0, True, {}

        obs, reward, done, info = self._current_env.step(action)

        if timeout:
            done = True
            info["timeout"] = True

        return obs, reward, done, info

    def iter_tasks(self, max_tasks: Optional[int] = None) -> List[AgentBenchTask]:
        return self.tasks[:max_tasks] if max_tasks else self.tasks

    def run_episode(
        self,
        agent: Any,
        task: Optional[AgentBenchTask] = None,
        device: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Run a full episode with an agent.

        Returns:
            Dict with reward, success, num_steps, action_history
        """
        obs = self.reset(task)
        total_reward = 0.0
        action_history = []

        for step in range(self.max_episode_steps):
            try:
                action_dict, _ = agent.step(observation=obs, device=device)
                action_str = str(action_dict.get("action", "")) + " " + str(action_dict.get("value", action_dict.get("command", "")))
                action_history.append(action_str)

                obs, reward, done, info = self.step(action_str)
                total_reward += reward

                if done:
                    break
            except Exception as e:
                logger.warning(f"Agent step error: {e}")
                break

        success = total_reward > 0.0

        return {
            "reward": total_reward,
            "success": success,
            "num_steps": len(action_history),
            "action_history": action_history,
        }
