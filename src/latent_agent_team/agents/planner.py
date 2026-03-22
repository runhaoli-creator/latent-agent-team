"""
planner.py — Planner agent: decomposes tasks into sub-goals, orchestrates team.

The Planner is the "conductor" of the agent team:
  1. Receives raw task instruction + environment observation
  2. Maintains a plan (ordered list of sub-goals)
  3. Decides which agent handles each sub-goal
  4. Routes sub-goals + latent context to appropriate agents
  5. Tracks plan progress and revises on failure

Uncertainty signals (used by AdaptiveBitrateScheduler):
  - planner_entropy: entropy of action distribution over plan steps
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from ..models.backbone import BackboneManager
from ..models.role_adapter import RoleAdapterManager
from ..models.latent_comm import LatentCommunicationModule, CommModuleConfig
from ..models.router import AdaptiveBitrateScheduler, SparseRouter

logger = logging.getLogger(__name__)


@dataclass
class PlanStep:
    step_id: int
    sub_goal: str
    assigned_agent: str
    status: str = "pending"   # pending | running | done | failed
    evidence: Optional[str] = None
    result: Optional[str] = None


@dataclass
class PlannerState:
    task_instruction: str
    steps: List[PlanStep] = field(default_factory=list)
    current_step_idx: int = 0
    completed: bool = False
    failure_count: int = 0
    last_entropy: float = 0.5


class PlannerAgent:
    """
    Planner agent using the shared backbone + role-specific LoRA + summarizer.
    """

    ROLE_NAME = "planner"

    def __init__(
        self,
        backbone: BackboneManager,
        role_manager: RoleAdapterManager,
        comm_module: LatentCommunicationModule,
        bitrate_scheduler: AdaptiveBitrateScheduler,
        sparse_router: SparseRouter,
        agent_idx: int = 0,
        max_plan_steps: int = 10,
        temperature: float = 0.7,
        max_new_tokens: int = 512,
    ):
        self.backbone = backbone
        self.role_manager = role_manager
        self.role_adapter = role_manager.get_adapter(self.ROLE_NAME)
        self.comm_module = comm_module
        self.bitrate_scheduler = bitrate_scheduler
        self.sparse_router = sparse_router
        self.agent_idx = agent_idx
        self.max_plan_steps = max_plan_steps
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.state: Optional[PlannerState] = None

    def reset(self, task_instruction: str) -> None:
        """Initialize planner for a new episode."""
        self.state = PlannerState(task_instruction=task_instruction)

    def _build_prompt(
        self,
        observation: str,
        incoming_messages: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Construct full input prompt for the planner."""
        role_prompt = self.role_adapter.get_role_prompt()
        parts = [
            f"[SYSTEM]\n{role_prompt}\n",
            f"[TASK]\n{self.state.task_instruction}\n",
            f"[OBSERVATION]\n{observation}\n",
        ]
        if self.state.steps:
            plan_text = "\n".join(
                f"  Step {s.step_id}: [{s.status}] {s.sub_goal} (agent={s.assigned_agent})"
                for s in self.state.steps
            )
            parts.append(f"[CURRENT PLAN]\n{plan_text}\n")
        if incoming_messages:
            msg_text = "\n".join(
                f"  [{m.get('sender', '?')}]: {m.get('content', '')}"
                for m in incoming_messages
            )
            parts.append(f"[MESSAGES FROM TEAM]\n{msg_text}\n")
        parts.append("[ACTION]")
        return "\n".join(parts)

    def _parse_action(self, generated_text: str) -> Dict[str, Any]:
        """Parse JSON action from generated text."""
        try:
            # Find first JSON object in generated text
            start = generated_text.find("{")
            end = generated_text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(generated_text[start:end])
        except (json.JSONDecodeError, ValueError):
            pass
        # Fallback
        return {
            "action": "plan",
            "sub_goals": [generated_text[:200]],
            "next_agent": "browser",
            "reasoning": generated_text[:100],
        }

    def _compute_entropy(self, logits: torch.Tensor) -> float:
        """Compute token distribution entropy from last-step logits."""
        probs = torch.softmax(logits[:, -1, :], dim=-1)
        entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1).mean()
        return float(entropy.item()) / 10.0  # normalize to ~[0,1]

    @torch.no_grad()
    def step(
        self,
        observation: str,
        incoming_latent: Optional[torch.Tensor] = None,  # latent from other agents
        incoming_messages: Optional[List[Dict[str, str]]] = None,  # text messages (baseline)
        device: Optional[torch.device] = None,
    ) -> Tuple[Dict[str, Any], torch.Tensor, float]:
        """
        Take one planning step.

        Returns:
            action:   Parsed plan/action dict
            hidden_states: Final-layer hidden states for latent communication
            entropy:  Planner's uncertainty estimate (for bitrate scheduler)
        """
        if self.state is None:
            raise RuntimeError("Call reset() before step()")

        prompt = self._build_prompt(observation, incoming_messages)

        # ── Tokenize ─────────────────────────────────────────────────────
        tokenizer = self.backbone.tokenizer
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.backbone.cfg.max_seq_len - self.max_new_tokens,
        )
        if device is not None:
            inputs = {k: v.to(device) for k, v in inputs.items()}

        # ── Set active adapter ───────────────────────────────────────────
        self.backbone.set_active_adapter(self.ROLE_NAME)

        # ── Inject incoming latent (if any) ──────────────────────────────
        if incoming_latent is not None and self.comm_module.mode != "text":
            # Get input embeddings, inject latent prefix
            embeds = self.backbone.model.get_input_embeddings()(inputs["input_ids"])
            combined_embeds, combined_mask, _ = self.comm_module(
                hidden_summary=incoming_latent,
                obs_embeds=embeds,
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

        # ── Extract generated text ───────────────────────────────────────
        new_ids = gen_outputs.sequences[:, inputs["input_ids"].shape[1]:]
        generated_text = tokenizer.decode(new_ids[0], skip_special_tokens=True)

        # ── Extract hidden states for outgoing latent comm ───────────────
        # Use hidden states from the last generated token step
        if hasattr(gen_outputs, "hidden_states") and gen_outputs.hidden_states:
            last_step_hidden = gen_outputs.hidden_states[-1][-1]  # [B, 1, hidden_size]
        else:
            # Fallback: run forward pass for hidden states
            with torch.no_grad():
                fwd_out = self.backbone.model(
                    **inputs, output_hidden_states=True
                )
            last_step_hidden = fwd_out.hidden_states[-1]  # [B, seq_len, hidden_size]

        # ── Compute entropy ──────────────────────────────────────────────
        with torch.no_grad():
            fwd_logits = self.backbone.model(**inputs).logits
        entropy = self._compute_entropy(fwd_logits)
        self.state.last_entropy = entropy

        # ── Parse action ─────────────────────────────────────────────────
        action = self._parse_action(generated_text)

        # ── Update plan state ────────────────────────────────────────────
        if "sub_goals" in action:
            for idx, sg in enumerate(action["sub_goals"][: self.max_plan_steps]):
                agent = action.get("next_agent", "browser")
                if idx >= len(self.state.steps):
                    self.state.steps.append(
                        PlanStep(
                            step_id=idx,
                            sub_goal=sg,
                            assigned_agent=agent,
                        )
                    )

        return action, last_step_hidden, entropy

    def generate_latent_message(
        self,
        hidden_states: torch.Tensor,
        k: int = 8,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Produce K latent tokens from planner's hidden states for outgoing messages.
        """
        return self.role_adapter.summarize_hidden_states(
            hidden_states, attention_mask, k=k
        )

    def mark_step_done(self, step_idx: int, result: str, success: bool) -> None:
        if step_idx < len(self.state.steps):
            self.state.steps[step_idx].status = "done" if success else "failed"
            self.state.steps[step_idx].result = result
            if not success:
                self.state.failure_count += 1

    def is_task_complete(self) -> bool:
        if self.state is None:
            return False
        done = all(s.status in ("done", "failed") for s in self.state.steps)
        success = all(s.status == "done" for s in self.state.steps)
        return done and success
