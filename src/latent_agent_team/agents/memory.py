"""
memory.py — Memory Manager agent: episodic memory + experience compression.

Responsibilities:
  1. Maintain a structured episodic memory of (obs, action, reward) triplets
  2. Compress and index experiences for efficient future retrieval
  3. Proactively surface relevant memories when team encounters similar situations
  4. Summarize long trajectories into compact representations
  5. Detect failure patterns from memory to guide recovery
"""

from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ..models.backbone import BackboneManager
from ..models.role_adapter import RoleAdapterManager
from ..models.latent_comm import LatentCommunicationModule

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    step: int
    observation: str
    action: Dict[str, Any]
    reward: float
    is_failure: bool
    sender_role: str
    episode_id: str
    embedding: Optional[np.ndarray] = None
    summary: Optional[str] = None


class EpisodicMemoryStore:
    """
    Ring-buffer episodic memory with optional FAISS-backed retrieval
    for long-horizon tasks.
    """

    def __init__(self, max_entries: int = 1000):
        self.max_entries = max_entries
        self.entries: deque[MemoryEntry] = deque(maxlen=max_entries)
        self._total_added = 0

    def add(self, entry: MemoryEntry) -> None:
        self.entries.append(entry)
        self._total_added += 1

    def get_recent(self, n: int = 10) -> List[MemoryEntry]:
        return list(self.entries)[-n:]

    def get_failures(self, n: int = 5) -> List[MemoryEntry]:
        failures = [e for e in self.entries if e.is_failure]
        return failures[-n:]

    def get_by_reward(self, threshold: float = 0.5, n: int = 5) -> List[MemoryEntry]:
        good = [e for e in self.entries if e.reward >= threshold]
        return good[-n:]

    def summarize_episode(self, episode_id: str) -> str:
        ep_entries = [e for e in self.entries if e.episode_id == episode_id]
        if not ep_entries:
            return ""
        total_reward = sum(e.reward for e in ep_entries)
        n_failures = sum(1 for e in ep_entries if e.is_failure)
        actions = [e.action.get("action", "?") for e in ep_entries[:10]]
        return (
            f"Episode {episode_id}: {len(ep_entries)} steps, "
            f"total_reward={total_reward:.2f}, failures={n_failures}, "
            f"actions=[{', '.join(actions[:5])}{'...' if len(actions) > 5 else ''}]"
        )

    def failure_pattern_analysis(self) -> Dict[str, int]:
        """Count failure types for debugging."""
        pattern_counts: Dict[str, int] = {}
        for entry in self.entries:
            if entry.is_failure:
                action_type = entry.action.get("action", "unknown")
                pattern_counts[action_type] = pattern_counts.get(action_type, 0) + 1
        return pattern_counts

    def to_jsonl_records(self) -> List[Dict]:
        records = []
        for e in self.entries:
            records.append({
                "step": e.step,
                "episode_id": e.episode_id,
                "observation": e.observation[:200],
                "action": e.action,
                "reward": e.reward,
                "is_failure": e.is_failure,
                "sender_role": e.sender_role,
                "summary": e.summary,
            })
        return records


class MemoryManager:
    """
    Memory Manager agent: maintains team memory and surfaces relevant experiences.
    """

    ROLE_NAME = "memory"

    def __init__(
        self,
        backbone: BackboneManager,
        role_manager: RoleAdapterManager,
        comm_module: LatentCommunicationModule,
        agent_idx: int = 4,
        max_memory_entries: int = 500,
        max_summary_length: int = 200,
        temperature: float = 0.5,
        max_new_tokens: int = 256,
        compression_interval: int = 20,  # compress every N steps
    ):
        self.backbone = backbone
        self.role_manager = role_manager
        self.role_adapter = role_manager.get_adapter(self.ROLE_NAME)
        self.comm_module = comm_module
        self.agent_idx = agent_idx
        self.max_summary_length = max_summary_length
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.compression_interval = compression_interval

        self.memory_store = EpisodicMemoryStore(max_entries=max_memory_entries)
        self._step_counter = 0
        self._current_episode_id = "ep_0000"

    def reset(self, episode_id: str) -> None:
        self._current_episode_id = episode_id
        self._step_counter = 0

    def store(
        self,
        step: int,
        observation: str,
        action: Dict[str, Any],
        reward: float,
        sender_role: str,
        is_failure: bool = False,
    ) -> None:
        """Store a trajectory step in memory."""
        entry = MemoryEntry(
            step=step,
            observation=observation,
            action=action,
            reward=reward,
            is_failure=is_failure,
            sender_role=sender_role,
            episode_id=self._current_episode_id,
        )
        self.memory_store.add(entry)
        self._step_counter += 1

        # Periodic compression
        if self._step_counter % self.compression_interval == 0:
            self._compress_recent_memory()

    def _compress_recent_memory(self) -> None:
        """Compress recent memory entries into summaries using the backbone."""
        recent = self.memory_store.get_recent(n=self.compression_interval)
        if not recent:
            return

        # Create a summary prompt
        obs_snippets = "\n".join(
            f"Step {e.step} [{e.sender_role}] {e.action.get('action', '?')}: "
            f"reward={e.reward:.2f}"
            for e in recent[:10]
        )
        prompt = (
            f"Summarize this agent trajectory in 1-2 sentences focusing on "
            f"what worked and what failed:\n{obs_snippets}\nSummary:"
        )

        tokenizer = self.backbone.tokenizer
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        device = self.backbone.device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        self.backbone.set_active_adapter(self.ROLE_NAME)
        with torch.no_grad():
            gen = self.backbone.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
            )
        new_ids = gen[:, inputs["input_ids"].shape[1]:]
        summary = tokenizer.decode(new_ids[0], skip_special_tokens=True)[:self.max_summary_length]

        # Store summary in latest entry
        if recent:
            recent[-1].summary = summary

    def recall(
        self,
        query: str,
        n: int = 5,
        include_failures: bool = True,
    ) -> List[MemoryEntry]:
        """
        Retrieve relevant memories for a query.
        Simple keyword-based for now; upgrades to FAISS if embeddings available.
        """
        recent = self.memory_store.get_recent(n=50)
        query_lower = query.lower()

        scored: List[Tuple[float, MemoryEntry]] = []
        for entry in recent:
            obs_score = sum(1 for w in query_lower.split() if w in entry.observation.lower())
            action_score = sum(
                1 for w in query_lower.split()
                if w in str(entry.action).lower()
            )
            failure_bonus = 0.5 if (include_failures and entry.is_failure) else 0.0
            score = obs_score + action_score + failure_bonus
            scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:n]]

    def _build_prompt(
        self,
        query: str,
        relevant_memories: List[MemoryEntry],
        request_type: str = "recall",
    ) -> str:
        role_prompt = self.role_adapter.get_role_prompt()
        parts = [f"[SYSTEM]\n{role_prompt}\n"]
        parts.append(f"[REQUEST]\n{request_type}: {query}\n")

        if relevant_memories:
            mem_text = "\n".join(
                f"  Step {m.step} [{m.sender_role}] {str(m.action)[:100]} "
                f"reward={m.reward:.2f} {'[FAILED]' if m.is_failure else ''}"
                for m in relevant_memories
            )
            parts.append(f"[RELEVANT MEMORIES]\n{mem_text}\n")

        failure_patterns = self.memory_store.failure_pattern_analysis()
        if failure_patterns:
            fp_text = ", ".join(f"{k}: {v}" for k, v in failure_patterns.items())
            parts.append(f"[FAILURE PATTERNS]\n{fp_text}\n")

        parts.append("[MEMORY RESPONSE]")
        return "\n".join(parts)

    @torch.no_grad()
    def step(
        self,
        query: str,
        request_type: str = "recall",
        incoming_latent: Optional[torch.Tensor] = None,
        incoming_messages: Optional[List[Dict]] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[Dict[str, Any], torch.Tensor]:
        """
        Process a memory query and generate a response.

        Returns:
            action: Memory response dict
            hidden_states: For outgoing latent messages
        """
        relevant = self.recall(query, n=5)
        prompt = self._build_prompt(query, relevant, request_type)

        tokenizer = self.backbone.tokenizer
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.backbone.cfg.max_seq_len - self.max_new_tokens,
        )
        if device is not None:
            inputs = {k: v.to(device) for k, v in inputs.items()}

        self.backbone.set_active_adapter(self.ROLE_NAME)

        if incoming_latent is not None and self.comm_module.mode != "text":
            embeds = self.backbone.model.get_input_embeddings()(inputs["input_ids"])
            combined_embeds, combined_mask, _ = self.comm_module(
                hidden_summary=incoming_latent, obs_embeds=embeds
            )
            gen_outputs = self.backbone.model.generate(
                inputs_embeds=combined_embeds,
                attention_mask=combined_mask,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
        else:
            gen_outputs = self.backbone.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )

        new_ids = gen_outputs.sequences[:, inputs["input_ids"].shape[1]:]
        generated_text = tokenizer.decode(new_ids[0], skip_special_tokens=True)

        if hasattr(gen_outputs, "hidden_states") and gen_outputs.hidden_states:
            hidden_states = gen_outputs.hidden_states[-1][-1]
        else:
            fwd_out = self.backbone.model(**inputs, output_hidden_states=True)
            hidden_states = fwd_out.hidden_states[-1]

        action = {
            "action": request_type,
            "query": query,
            "content": generated_text[:self.max_summary_length],
            "relevant_memories": len(relevant),
            "failure_patterns": self.memory_store.failure_pattern_analysis(),
        }

        return action, hidden_states

    def generate_latent_message(
        self,
        hidden_states: torch.Tensor,
        k: int = 8,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.role_adapter.summarize_hidden_states(hidden_states, attention_mask, k=k)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_entries": len(self.memory_store.entries),
            "total_failures": sum(1 for e in self.memory_store.entries if e.is_failure),
            "current_episode": self._current_episode_id,
            "failure_patterns": self.memory_store.failure_pattern_analysis(),
        }
