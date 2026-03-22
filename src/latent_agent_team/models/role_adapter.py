"""
role_adapter.py — Role-specific LoRA adapters + role prompt management.

Each agent role (Planner, Retriever, Browser, Verifier, Memory) gets:
  1. A system/role prompt template defining its behavior
  2. A lightweight LoRA adapter on the shared backbone
  3. A hidden-state summarizer for latent communication
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType

logger = logging.getLogger(__name__)

# ── Role definitions ─────────────────────────────────────────────────────────

ROLE_PROMPTS: Dict[str, str] = {
    "planner": (
        "You are the Planner agent. Your job is to decompose complex tasks into "
        "sequential sub-goals, decide which agent should handle each sub-goal, "
        "and maintain the overall task plan. You receive observations from the "
        "environment and messages from other agents, then emit the next sub-goal "
        "or revise the plan.\n"
        "Output format: {\"action\": \"plan\", \"sub_goals\": [...], \"next_agent\": \"...\", "
        "\"reasoning\": \"...\"}"
    ),
    "retriever": (
        "You are the Retriever agent. Your job is to search memory, retrieve "
        "relevant past experiences, documentation, or product information to "
        "support the current task. You receive queries from the Planner or "
        "Browser and return ranked evidence.\n"
        "Output format: {\"action\": \"retrieve\", \"query\": \"...\", "
        "\"results\": [...], \"confidence\": 0.0}"
    ),
    "browser": (
        "You are the Browser/Interactor agent. Your job is to interact with web "
        "environments: click elements, type text, navigate pages, select products, "
        "and extract structured information from DOM trees. You execute low-level "
        "actions based on the Planner's sub-goals.\n"
        "Output format: {\"action\": \"click|type|scroll|select|navigate\", "
        "\"element\": \"...\", \"value\": \"...\", \"reasoning\": \"...\"}"
    ),
    "verifier": (
        "You are the Verifier agent. Your job is to check whether the current "
        "state satisfies the task constraints, detect errors or contradictions, "
        "and decide whether to accept the current trajectory or request recovery. "
        "You compare the current state against the goal and past evidence.\n"
        "Output format: {\"action\": \"verify\", \"status\": \"pass|fail|uncertain\", "
        "\"issues\": [...], \"suggestion\": \"...\"}"
    ),
    "memory": (
        "You are the Memory Manager agent. Your job is to maintain an episodic "
        "memory of past observations, actions, and outcomes. You compress and "
        "index experiences for efficient retrieval, and proactively surface "
        "relevant memories when the team encounters similar situations.\n"
        "Output format: {\"action\": \"store|recall|summarize\", \"key\": \"...\", "
        "\"content\": \"...\", \"relevance\": 0.0}"
    ),
}


@dataclass
class RoleAdapterConfig:
    """Configuration for a role-specific LoRA adapter."""
    role_name: str = "planner"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: Optional[List[str]] = None
    # Hidden-state summarizer config
    summarizer_pool_size: int = 16  # N: pool last N task-relevant positions
    summarizer_hidden_dim: int = 256
    summarizer_num_layers: int = 2  # 2-layer Transformer bottleneck
    summarizer_num_heads: int = 4


class HiddenStateSummarizer(nn.Module):
    """
    Takes final-layer hidden states from the current turn, pools the last N
    task-relevant positions, passes them through a small projection MLP or
    2-layer Transformer bottleneck, and emits K latent message tokens in
    embedding space.

    This implements the core Interlat-style hidden-state compression.
    """

    def __init__(
        self,
        hidden_size: int,
        pool_size: int = 16,
        bottleneck_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        max_k: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.pool_size = pool_size
        self.bottleneck_dim = bottleneck_dim
        self.max_k = max_k

        # Project from backbone hidden size to bottleneck
        self.input_proj = nn.Linear(hidden_size, bottleneck_dim)
        self.input_norm = nn.LayerNorm(bottleneck_dim)

        # Learnable query tokens for latent messages (max K)
        self.latent_queries = nn.Parameter(
            torch.randn(max_k, bottleneck_dim) * 0.02
        )

        # 2-layer Transformer bottleneck (cross-attention from queries to hidden states)
        encoder_layer = nn.TransformerDecoderLayer(
            d_model=bottleneck_dim,
            nhead=num_heads,
            dim_feedforward=bottleneck_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_bottleneck = nn.TransformerDecoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Project back to backbone embedding space
        self.output_proj = nn.Linear(bottleneck_dim, hidden_size)
        self.output_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,  # [B, seq_len, hidden_size]
        attention_mask: Optional[torch.Tensor] = None,  # [B, seq_len]
        k: int = 8,  # number of latent tokens to emit
    ) -> torch.Tensor:
        """
        Produce K latent message tokens from the backbone's hidden states.

        Args:
            hidden_states: Final-layer hidden states [B, seq_len, hidden_size]
            attention_mask: Mask for valid positions [B, seq_len]
            k: Number of latent tokens to emit (adaptive bitrate)

        Returns:
            latent_tokens: [B, K, hidden_size] — latent message in embedding space
        """
        B = hidden_states.size(0)
        assert k <= self.max_k, f"k={k} exceeds max_k={self.max_k}"

        # Pool last N task-relevant positions
        if attention_mask is not None:
            # Find the last valid positions per batch
            lengths = attention_mask.sum(dim=1).long()  # [B]
            pooled = []
            for i in range(B):
                end = lengths[i].item()
                start = max(0, end - self.pool_size)
                chunk = hidden_states[i, start:end, :]  # [N', hidden_size]
                if chunk.size(0) < self.pool_size:
                    pad = torch.zeros(
                        self.pool_size - chunk.size(0),
                        self.hidden_size,
                        device=chunk.device,
                        dtype=chunk.dtype,
                    )
                    chunk = torch.cat([pad, chunk], dim=0)
                pooled.append(chunk)
            pooled = torch.stack(pooled, dim=0)  # [B, pool_size, hidden_size]
        else:
            pooled = hidden_states[:, -self.pool_size :, :]  # [B, N, hidden_size]

        # Project to bottleneck dimension
        memory = self.input_norm(self.input_proj(pooled))  # [B, N, bottleneck_dim]

        # Expand learnable queries for the requested K
        queries = self.latent_queries[:k].unsqueeze(0).expand(B, -1, -1)  # [B, K, bottleneck_dim]

        # Cross-attend: queries attend to hidden states through Transformer decoder
        latent = self.transformer_bottleneck(queries, memory)  # [B, K, bottleneck_dim]

        # Project back to backbone embedding space
        latent_tokens = self.output_norm(self.output_proj(latent))  # [B, K, hidden_size]

        return latent_tokens


class RoleAdapter(nn.Module):
    """
    Wraps a role-specific LoRA config + hidden-state summarizer for one agent.
    """

    def __init__(
        self,
        role_name: str,
        hidden_size: int,
        config: Optional[RoleAdapterConfig] = None,
    ):
        super().__init__()
        if config is None:
            config = RoleAdapterConfig(role_name=role_name)

        self.role_name = role_name
        self.config = config
        self.role_prompt = ROLE_PROMPTS.get(role_name, "You are a helpful assistant.")

        # LoRA config (to be applied to backbone)
        self.lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.target_modules,
            bias="none",
        )

        # Hidden-state summarizer
        self.summarizer = HiddenStateSummarizer(
            hidden_size=hidden_size,
            pool_size=config.summarizer_pool_size,
            bottleneck_dim=config.summarizer_hidden_dim,
            num_layers=config.summarizer_num_layers,
            num_heads=config.summarizer_num_heads,
        )

    def get_role_prompt(self) -> str:
        return self.role_prompt

    def summarize_hidden_states(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        k: int = 8,
    ) -> torch.Tensor:
        """Produce K latent tokens from the agent's hidden states."""
        return self.summarizer(hidden_states, attention_mask, k)


class RoleAdapterManager:
    """
    Manages all role adapters and their summarizers.
    """

    def __init__(self, hidden_size: int, roles: Optional[List[str]] = None):
        self.hidden_size = hidden_size
        self.roles = roles or ["planner", "retriever", "browser", "verifier", "memory"]
        self.adapters: Dict[str, RoleAdapter] = {}

        for role in self.roles:
            self.adapters[role] = RoleAdapter(
                role_name=role,
                hidden_size=hidden_size,
            )
            logger.info(f"Initialized RoleAdapter for '{role}'")

    def get_adapter(self, role_name: str) -> RoleAdapter:
        if role_name not in self.adapters:
            raise ValueError(f"Unknown role: {role_name}")
        return self.adapters[role_name]

    def get_all_summarizer_params(self):
        """Yield all trainable summarizer parameters across roles."""
        for role_name, adapter in self.adapters.items():
            for name, param in adapter.summarizer.named_parameters():
                yield f"{role_name}.summarizer.{name}", param

    def to(self, device: torch.device):
        for adapter in self.adapters.values():
            adapter.summarizer.to(device)
        return self
