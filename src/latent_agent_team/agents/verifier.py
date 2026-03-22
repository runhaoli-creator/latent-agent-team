"""
verifier.py — Verifier agent: constraint checking, disagreement detection, recovery.

Responsibilities:
  1. Check whether the current state satisfies task constraints
  2. Detect contradictions between evidence and current action
  3. Score verifier confidence (used by AdaptiveBitrateScheduler)
  4. Provide recovery suggestions on failure
  5. Supply verifier labels for training data (pass/fail/uncertain)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from ..models.backbone import BackboneManager
from ..models.role_adapter import RoleAdapterManager
from ..models.latent_comm import LatentCommunicationModule

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    status: str  # "pass" | "fail" | "uncertain"
    confidence: float
    issues: List[str]
    suggestion: str
    disagreement_score: float  # 0-1 disagreement with proposed action


class VerifierAgent:
    """
    Verifier agent: validates task progress and detects failures.
    """

    ROLE_NAME = "verifier"

    def __init__(
        self,
        backbone: BackboneManager,
        role_manager: RoleAdapterManager,
        comm_module: LatentCommunicationModule,
        agent_idx: int = 3,
        temperature: float = 0.1,  # low temp for deterministic verification
        max_new_tokens: int = 256,
        disagreement_threshold: float = 0.6,
    ):
        self.backbone = backbone
        self.role_manager = role_manager
        self.role_adapter = role_manager.get_adapter(self.ROLE_NAME)
        self.comm_module = comm_module
        self.agent_idx = agent_idx
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.disagreement_threshold = disagreement_threshold
        self.verification_history: List[VerificationResult] = []

    def _build_prompt(
        self,
        observation: str,
        task_goal: str,
        proposed_action: Optional[str] = None,
        past_verifications: Optional[List[VerificationResult]] = None,
        incoming_messages: Optional[List[Dict]] = None,
    ) -> str:
        role_prompt = self.role_adapter.get_role_prompt()
        parts = [f"[SYSTEM]\n{role_prompt}\n"]
        parts.append(f"[TASK GOAL]\n{task_goal}\n")
        parts.append(f"[CURRENT OBSERVATION]\n{observation[:600]}\n")

        if proposed_action:
            parts.append(f"[PROPOSED ACTION]\n{proposed_action}\n")

        if past_verifications:
            recent = past_verifications[-3:]
            hist = "\n".join(
                f"  [{v.status}] conf={v.confidence:.2f}: {', '.join(v.issues[:2])}"
                for v in recent
            )
            parts.append(f"[PAST VERIFICATIONS]\n{hist}\n")

        if incoming_messages:
            msg_text = "\n".join(
                f"  [{m.get('sender', '?')}]: {m.get('content', '')[:200]}"
                for m in incoming_messages
            )
            parts.append(f"[TEAM MESSAGES]\n{msg_text}\n")

        parts.append("[VERIFICATION]")
        return "\n".join(parts)

    def _parse_verification(self, text: str) -> VerificationResult:
        """Parse verification output from model."""
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                d = json.loads(text[start:end])
                status = d.get("status", "uncertain")
                if status not in ("pass", "fail", "uncertain"):
                    status = "uncertain"
                return VerificationResult(
                    status=status,
                    confidence=float(d.get("confidence", 0.5)),
                    issues=d.get("issues", []),
                    suggestion=d.get("suggestion", ""),
                    disagreement_score=0.0,
                )
        except (json.JSONDecodeError, ValueError, KeyError):
            pass

        # Keyword heuristic fallback
        text_lower = text.lower()
        if any(w in text_lower for w in ["pass", "correct", "success", "valid", "done"]):
            status, conf = "pass", 0.7
        elif any(w in text_lower for w in ["fail", "wrong", "incorrect", "error", "missing"]):
            status, conf = "fail", 0.7
        else:
            status, conf = "uncertain", 0.4

        return VerificationResult(
            status=status,
            confidence=conf,
            issues=[text[:100]],
            suggestion="",
            disagreement_score=0.0,
        )

    @torch.no_grad()
    def step(
        self,
        observation: str,
        task_goal: str,
        proposed_action: Optional[str] = None,
        incoming_latent: Optional[torch.Tensor] = None,
        incoming_messages: Optional[List[Dict]] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[VerificationResult, torch.Tensor]:
        """
        Verify the current state against task goal.

        Returns:
            result: VerificationResult with status, confidence, issues
            hidden_states: For outgoing latent messages
        """
        prompt = self._build_prompt(
            observation, task_goal, proposed_action,
            self.verification_history[-3:], incoming_messages
        )

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
                do_sample=False,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
        else:
            gen_outputs = self.backbone.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=False,
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

        result = self._parse_verification(generated_text)
        self.verification_history.append(result)

        return result, hidden_states

    def compute_disagreement_score(
        self,
        verification_results: List[VerificationResult],
    ) -> float:
        """
        Compute disagreement from recent verifications.
        High disagreement = verifiers conflict = hard step → more latent budget.
        """
        if len(verification_results) < 2:
            return 0.0
        statuses = [v.status for v in verification_results[-3:]]
        pass_count = statuses.count("pass")
        fail_count = statuses.count("fail")
        total = len(statuses)
        disagreement = (min(pass_count, fail_count) / total) * 2
        return disagreement

    @property
    def last_disagreement(self) -> float:
        return self.compute_disagreement_score(self.verification_history)

    def generate_latent_message(
        self,
        hidden_states: torch.Tensor,
        k: int = 8,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.role_adapter.summarize_hidden_states(hidden_states, attention_mask, k=k)

    def reset(self) -> None:
        self.verification_history = []
