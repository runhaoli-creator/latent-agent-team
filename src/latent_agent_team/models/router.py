"""
router.py — Sparse routing + adaptive bitrate scheduling for latent messages.

AdaptiveBitrateScheduler:
  Predicts K ∈ {4, 8, 16, 32, 64} from uncertainty features:
    - planner_entropy  (high → hard task)
    - verifier_disagreement (high → conflicting evidence)
    - retrieval_confidence  (low → harder to find evidence)
    - recent_failure_count  (high → recovery needed)
  Easy steps → K = 4–8; Hard steps → K = 32–64.

SparseRouter:
  Decides which subset of N agents receives each message.
  Output: binary routing mask [num_agents] per sender message.
  Loss: routing supervision from teacher traces.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Allowed K values (bits per message)
K_CHOICES = [4, 8, 16, 32, 64]
NUM_AGENTS = 5  # planner, retriever, browser, verifier, memory
AGENT_NAMES = ["planner", "retriever", "browser", "verifier", "memory"]


class AdaptiveBitrateScheduler(nn.Module):
    """
    Predicts the optimal K (number of latent tokens) from uncertainty features.

    Input features (all scalar, passed as a single vector):
        [planner_entropy, verifier_disagreement, retrieval_confidence_inv,
         recent_failure_rate, task_progress, step_budget_remaining]

    Output: distribution over K_CHOICES (soft during training, argmax at inference).
    """

    INPUT_DIM = 6  # number of uncertainty features

    def __init__(
        self,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        entropy_threshold_low: float = 0.3,
        entropy_threshold_high: float = 0.7,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.k_choices = K_CHOICES
        self.num_k = len(K_CHOICES)
        self.entropy_threshold_low = entropy_threshold_low
        self.entropy_threshold_high = entropy_threshold_high

        # Small MLP for K prediction
        self.predictor = nn.Sequential(
            nn.Linear(self.INPUT_DIM, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_k),
        )

        # Learnable temperature for K selection (anneals toward hard choice)
        self.log_temperature = nn.Parameter(torch.zeros(1))

    @property
    def temperature(self) -> torch.Tensor:
        return F.softplus(self.log_temperature) + 0.1

    def _build_features(
        self,
        planner_entropy: float = 0.5,
        verifier_disagreement: float = 0.0,
        retrieval_confidence: float = 1.0,
        recent_failure_count: int = 0,
        task_progress: float = 0.0,
        step_budget_remaining: float = 1.0,
    ) -> torch.Tensor:
        """Build the feature vector on CPU (moved to device in forward)."""
        feat = torch.tensor(
            [
                float(planner_entropy),
                float(verifier_disagreement),
                1.0 - float(retrieval_confidence),  # inverted: high = hard
                min(1.0, float(recent_failure_count) / 5.0),
                float(task_progress),
                float(step_budget_remaining),
            ],
            dtype=torch.float32,
        )
        return feat

    def forward(
        self,
        features: torch.Tensor,  # [B, INPUT_DIM] or [INPUT_DIM]
        hard: bool = False,       # True at inference time
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: Uncertainty features [B, INPUT_DIM] or [INPUT_DIM]
            hard:     If True, return one-hot argmax; else soft probabilities

        Returns:
            k_probs:  [B, num_k] — soft distribution over K choices
            k_values: [B] — selected K values (integers)
        """
        if features.dim() == 1:
            features = features.unsqueeze(0)  # [1, INPUT_DIM]

        logits = self.predictor(features)  # [B, num_k]

        if hard:
            k_idx = logits.argmax(dim=-1)   # [B]
            k_values = torch.tensor(
                [self.k_choices[i.item()] for i in k_idx],
                device=features.device,
            )
            k_probs = F.one_hot(k_idx, num_classes=self.num_k).float()
        else:
            k_probs = F.softmax(logits / self.temperature, dim=-1)  # [B, num_k]
            k_idx = k_probs.argmax(dim=-1)
            k_values = torch.tensor(
                [self.k_choices[i.item()] for i in k_idx],
                device=features.device,
            )

        return k_probs, k_values

    def select_k(
        self,
        planner_entropy: float = 0.5,
        verifier_disagreement: float = 0.0,
        retrieval_confidence: float = 1.0,
        recent_failure_count: int = 0,
        task_progress: float = 0.0,
        step_budget_remaining: float = 1.0,
        device: Optional[torch.device] = None,
        hard: bool = True,
    ) -> int:
        """Convenience method: select K from scalar uncertainty features."""
        feat = self._build_features(
            planner_entropy=planner_entropy,
            verifier_disagreement=verifier_disagreement,
            retrieval_confidence=retrieval_confidence,
            recent_failure_count=recent_failure_count,
            task_progress=task_progress,
            step_budget_remaining=step_budget_remaining,
        )
        if device is not None:
            feat = feat.to(device)
        _, k_values = self.forward(feat, hard=hard)
        return int(k_values[0].item())

    def bitrate_regularization_loss(
        self,
        k_probs: torch.Tensor,   # [B, num_k]
        lambda_bitrate: float = 0.01,
    ) -> torch.Tensor:
        """
        Penalize high-K choices to encourage efficient communication.
        L_br = λ * E[K] = λ * Σ_k p(k) * k_value
        """
        k_vals = torch.tensor(
            self.k_choices, dtype=torch.float32, device=k_probs.device
        )
        expected_k = (k_probs * k_vals).sum(dim=-1).mean()
        return lambda_bitrate * expected_k


class SparseRouter(nn.Module):
    """
    Determines which subset of N agents should receive each latent message.
    Replaces full broadcast with targeted delivery for communication efficiency.

    For each (sender, message) pair, produces a binary routing mask over agents.
    Trained with routing supervision from teacher traces.

    Features per routing decision:
        - sender role embedding
        - message content features (aggregated from latent tokens)
        - recipient role embedding
        - step context features
    """

    def __init__(
        self,
        hidden_size: int,
        num_agents: int = NUM_AGENTS,
        agent_names: List[str] = AGENT_NAMES,
        router_hidden_dim: int = 128,
        dropout: float = 0.1,
        routing_threshold: float = 0.5,
        min_recipients: int = 1,
        max_recipients: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_agents = num_agents
        self.agent_names = agent_names
        self.routing_threshold = routing_threshold
        self.min_recipients = min_recipients
        self.max_recipients = max_recipients or num_agents

        # Learnable role embeddings (sender and candidate recipient)
        self.role_embeddings = nn.Embedding(num_agents, router_hidden_dim)

        # Message content projection (from K latent tokens → scalar summary)
        self.message_proj = nn.Sequential(
            nn.Linear(hidden_size, router_hidden_dim),
            nn.GELU(),
        )

        # Per-pair routing score head
        # Input: [sender_emb | recipient_emb | message_summary]
        self.routing_head = nn.Sequential(
            nn.Linear(router_hidden_dim * 3, router_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(router_hidden_dim, 1),
        )

    def forward(
        self,
        sender_idx: int,
        latent_tokens: torch.Tensor,      # [B, K, hidden_size]
        recipient_mask: Optional[torch.Tensor] = None,  # [num_agents] allowed recipients
        hard: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute routing scores for each candidate recipient.

        Args:
            sender_idx:    Index of the sending agent
            latent_tokens: Latent message tokens [B, K, hidden_size]
            recipient_mask: Boolean mask of allowed recipients [num_agents]
            hard:          If True, threshold to binary mask

        Returns:
            routing_probs:  [B, num_agents] — soft routing probabilities
            routing_mask:   [B, num_agents] — binary routing decisions
        """
        B = latent_tokens.size(0)
        device = latent_tokens.device

        # Sender role embedding [B, router_hidden_dim]
        sender_emb = self.role_embeddings(
            torch.tensor(sender_idx, device=device).expand(B)
        )

        # Message summary: mean-pool latent tokens, then project
        msg_summary = latent_tokens.mean(dim=1)  # [B, hidden_size]
        msg_features = self.message_proj(msg_summary)  # [B, router_hidden_dim]

        # Score each possible recipient
        all_scores = []
        for r_idx in range(self.num_agents):
            if r_idx == sender_idx:
                # Agent doesn't send to itself
                all_scores.append(torch.full((B,), -1e9, device=device))
                continue

            recip_emb = self.role_embeddings(
                torch.tensor(r_idx, device=device).expand(B)
            )  # [B, router_hidden_dim]

            pair_feat = torch.cat([sender_emb, recip_emb, msg_features], dim=-1)
            score = self.routing_head(pair_feat).squeeze(-1)  # [B]
            all_scores.append(score)

        scores = torch.stack(all_scores, dim=-1)  # [B, num_agents]

        # Apply optional mask (e.g., don't route to inactive agents)
        if recipient_mask is not None:
            mask_f = recipient_mask.float().to(device)
            scores = scores * mask_f + (1 - mask_f) * (-1e9)

        routing_probs = torch.sigmoid(scores)  # [B, num_agents]

        if hard:
            routing_mask = (routing_probs >= self.routing_threshold).float()
            # Ensure at least min_recipients
            if routing_mask.sum(dim=-1).min() < self.min_recipients:
                top_k = routing_probs.topk(self.min_recipients, dim=-1).indices
                routing_mask.scatter_(-1, top_k, 1.0)
        else:
            routing_mask = routing_probs  # soft for training

        return routing_probs, routing_mask

    def routing_supervision_loss(
        self,
        routing_probs: torch.Tensor,     # [B, num_agents]
        teacher_routing: torch.Tensor,   # [B, num_agents] — binary targets from teacher
    ) -> torch.Tensor:
        """Binary cross-entropy loss against teacher routing decisions."""
        return F.binary_cross_entropy(
            routing_probs.clamp(1e-7, 1 - 1e-7),
            teacher_routing.float(),
            reduction="mean",
        )

    def get_recipient_names(
        self, routing_mask: torch.Tensor, batch_idx: int = 0
    ) -> List[str]:
        """Get names of selected recipients for a given batch element."""
        selected = routing_mask[batch_idx].bool()
        return [self.agent_names[i] for i in range(self.num_agents) if selected[i]]
