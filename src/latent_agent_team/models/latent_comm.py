"""
latent_comm.py — Latent communication channel between agents.

Implements two modes:
  1. ContinuousLatentChannel  — passes raw latent token embeddings
  2. VQLatentChannel          — vector-quantized tokens using learned codebook
     with commitment loss, EMA codebook updates, optional Gumbel-softmax warm start

Receivers consume latent tokens via a learned soft-prompt prefix or
cross-attention bridge before their next action.

Core idea from Interlat: transmit last-layer hidden states rather than text,
showing that compressed latent communication preserves downstream utility.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Vector Quantizer (VQ-VAE style with EMA updates + Gumbel-softmax warmup)
# ─────────────────────────────────────────────────────────────────────────────

class VectorQuantizer(nn.Module):
    """
    Learned codebook with:
    - EMA codebook updates (stable, no encoder collapse)
    - Commitment loss (encoder stays close to codebook)
    - Optional Gumbel-softmax warm start for initial training
    - Dead code restart (reset unused codebook entries from encoder outputs)
    - Entropy regularization to encourage codebook utilization

    Reference: van den Oord et al. (2017) "Neural Discrete Representation Learning"
    """

    def __init__(
        self,
        codebook_size: int = 512,
        embedding_dim: int = 256,
        commitment_cost: float = 0.25,
        ema_decay: float = 0.99,
        ema_epsilon: float = 1e-5,
        use_gumbel_warmup: bool = True,
        gumbel_temperature_start: float = 2.0,
        gumbel_temperature_end: float = 0.5,
        gumbel_anneal_steps: int = 10000,
        dead_code_threshold: int = 2,
        entropy_weight: float = 0.1,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.ema_decay = ema_decay
        self.ema_epsilon = ema_epsilon
        self.use_gumbel_warmup = use_gumbel_warmup
        self.gumbel_temp_start = gumbel_temperature_start
        self.gumbel_temp_end = gumbel_temperature_end
        self.gumbel_anneal_steps = gumbel_anneal_steps
        self.dead_code_threshold = dead_code_threshold
        self.entropy_weight = entropy_weight

        # Codebook embedding — initialized with larger range for better coverage
        self.codebook = nn.Embedding(codebook_size, embedding_dim)
        nn.init.normal_(self.codebook.weight, mean=0.0, std=1.0)

        # EMA statistics (not learnable parameters — updated manually)
        self.register_buffer("ema_cluster_size", torch.zeros(codebook_size))
        self.register_buffer("ema_dw", self.codebook.weight.data.clone())
        self.register_buffer("global_step", torch.tensor(0, dtype=torch.long))
        self.register_buffer("_initialized", torch.tensor(0, dtype=torch.long))
        # Track usage for dead code restart
        self.register_buffer("code_usage", torch.zeros(codebook_size, dtype=torch.long))

    @property
    def gumbel_temperature(self) -> float:
        """Linearly anneal Gumbel temperature from start to end."""
        step = self.global_step.item()
        ratio = min(1.0, step / max(1, self.gumbel_anneal_steps))
        return self.gumbel_temp_start + ratio * (self.gumbel_temp_end - self.gumbel_temp_start)

    def quantize(
        self,
        z: torch.Tensor,  # [B, K, embedding_dim]
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize continuous latent vectors to codebook indices.

        Returns:
            z_q:         quantized embeddings [B, K, embedding_dim]
            commit_loss: commitment + codebook loss (scalar)
            indices:     codebook indices [B, K]
        """
        B, K, D = z.shape
        z_flat = z.reshape(-1, D)  # [B*K, D]

        # K-means initialization from first batch of data
        if training and self._initialized.item() == 0 and z_flat.shape[0] >= self.codebook_size:
            logger.info("VQ: Initializing codebook from data (k-means style)")
            # Random selection with perturbation
            perm = torch.randperm(z_flat.shape[0])[:self.codebook_size]
            self.codebook.weight.data = z_flat[perm].clone()
            self.ema_dw.data = self.codebook.weight.data.clone()
            self.ema_cluster_size.fill_(1.0)
            self._initialized.fill_(1)
        elif training and self._initialized.item() == 0:
            # Not enough data yet — init from available data + noise
            n_avail = z_flat.shape[0]
            if n_avail > 0:
                repeats = (self.codebook_size // n_avail) + 1
                init_data = z_flat.repeat(repeats, 1)[:self.codebook_size]
                noise = torch.randn_like(init_data) * 0.01
                self.codebook.weight.data = init_data + noise
                self.ema_dw.data = self.codebook.weight.data.clone()
                self.ema_cluster_size.fill_(1.0)
                self._initialized.fill_(1)

        # Normalize both z and codebook for more stable distances
        z_flat_norm = F.normalize(z_flat, dim=-1)
        cb_norm = F.normalize(self.codebook.weight, dim=-1)

        # Compute pairwise distances to codebook (using normalized cosine-like distance)
        dist = (
            z_flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * z_flat @ self.codebook.weight.T
            + self.codebook.weight.pow(2).sum(dim=1)
        )  # [B*K, codebook_size]

        if self.use_gumbel_warmup and training and self.gumbel_temperature > self.gumbel_temp_end:
            # Gumbel-softmax for soft differentiation during warmup
            logits = -dist
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
            soft_indices = F.softmax((logits + gumbel_noise) / self.gumbel_temperature, dim=-1)
            z_q_flat = soft_indices @ self.codebook.weight  # [B*K, D]
            indices = soft_indices.argmax(dim=-1)  # [B*K]
        else:
            # Hard nearest-neighbor lookup
            indices = dist.argmin(dim=-1)  # [B*K]
            z_q_flat = self.codebook(indices)  # [B*K, D]

        # ── EMA codebook update ──────────────────────────────────────────
        if training:
            with torch.no_grad():
                one_hot = F.one_hot(
                    dist.argmin(dim=-1), num_classes=self.codebook_size
                ).float()  # [B*K, codebook_size]

                new_cluster_size = one_hot.sum(0)
                new_dw = one_hot.T @ z_flat  # [codebook_size, D]

                # EMA update
                self.ema_cluster_size = (
                    self.ema_decay * self.ema_cluster_size
                    + (1 - self.ema_decay) * new_cluster_size
                )
                self.ema_dw = (
                    self.ema_decay * self.ema_dw
                    + (1 - self.ema_decay) * new_dw
                )

                # Normalize to get updated codebook
                n = self.ema_cluster_size.sum()
                cluster_size_smoothed = (
                    (self.ema_cluster_size + self.ema_epsilon)
                    / (n + self.codebook_size * self.ema_epsilon)
                    * n
                )
                self.codebook.weight.data = self.ema_dw / cluster_size_smoothed.unsqueeze(1)

                # Track code usage
                self.code_usage += new_cluster_size.long()

                # Dead code restart: replace unused codebook entries with encoder outputs
                if self.global_step.item() % 100 == 0 and self.global_step.item() > 0:
                    dead_mask = self.code_usage < self.dead_code_threshold
                    n_dead = dead_mask.sum().item()
                    if n_dead > 0 and z_flat.shape[0] > 0:
                        # Replace dead codes with random encoder outputs + noise
                        replace_idx = torch.randperm(z_flat.shape[0])[:n_dead]
                        if len(replace_idx) > 0:
                            dead_indices = torch.where(dead_mask)[0][:len(replace_idx)]
                            self.codebook.weight.data[dead_indices] = z_flat[replace_idx].clone() + torch.randn(len(dead_indices), D, device=z_flat.device) * 0.01
                            self.ema_dw.data[dead_indices] = self.codebook.weight.data[dead_indices].clone()
                            self.ema_cluster_size[dead_indices] = 1.0
                            self.code_usage[dead_indices] = 0
                            logger.debug(f"VQ: Restarted {len(dead_indices)} dead codes at step {self.global_step.item()}")

            self.global_step += 1

        # ── Losses ───────────────────────────────────────────────────────
        z_q_reshaped = z_q_flat.reshape(B, K, D)

        # Commitment loss: encoder output stays close to chosen codebook entry
        commit_loss = F.mse_loss(z_q_reshaped.detach(), z) * self.commitment_cost
        # Codebook loss: not needed with EMA, but kept for straight-through gradient
        codebook_loss = F.mse_loss(z_q_reshaped, z.detach())

        # Entropy regularization: encourage uniform codebook usage
        avg_probs = F.softmax(-dist, dim=-1).mean(dim=0)  # [codebook_size]
        entropy = -(avg_probs * (avg_probs + 1e-10).log()).sum()
        max_entropy = math.log(self.codebook_size)
        entropy_loss = self.entropy_weight * (max_entropy - entropy) / max_entropy

        # Straight-through estimator: copy gradients from z_q to z
        z_q_st = z + (z_q_reshaped - z).detach()

        total_loss = commit_loss + codebook_loss + entropy_loss
        indices = indices.reshape(B, K)

        return z_q_st, total_loss, indices


# ─────────────────────────────────────────────────────────────────────────────
#  Latent Encoder / Decoder
# ─────────────────────────────────────────────────────────────────────────────

class LatentEncoder(nn.Module):
    """
    Projects hidden-state summaries to a lower-dimensional latent space
    before quantization or direct transmission.
    """

    def __init__(self, hidden_size: int, latent_dim: int = 256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: [B, K, hidden_size] → [B, K, latent_dim]"""
        return self.proj(h)


class LatentDecoder(nn.Module):
    """
    Projects latent tokens back to backbone embedding space so they can be
    prepended as a soft-prompt prefix for the receiving agent.
    """

    def __init__(self, latent_dim: int = 256, hidden_size: int = 3072):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, hidden_size),
            nn.LayerNorm(hidden_size),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: [B, K, latent_dim] → [B, K, hidden_size]"""
        return self.proj(z)


# ─────────────────────────────────────────────────────────────────────────────
#  Soft-Prompt Prefix Injector (receiver side)
# ─────────────────────────────────────────────────────────────────────────────

class SoftPromptPrefixInjector(nn.Module):
    """
    Prepends K decoded latent tokens as a soft-prompt prefix to the receiver's
    input embeddings.  The receiver's backbone then sees:
      [latent_prefix | observation_tokens]
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        # Scale gate: learned scalar gating for how much to trust latent prefix
        self.scale = nn.Parameter(torch.ones(1))

    def forward(
        self,
        obs_embeds: torch.Tensor,       # [B, seq_len, hidden_size]
        latent_prefix: torch.Tensor,    # [B, K, hidden_size]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            combined_embeds: [B, K + seq_len, hidden_size]
            prefix_mask:     [B, K + seq_len] — ones for prefix, then obs mask
        """
        gated_prefix = self.scale * latent_prefix
        combined = torch.cat([gated_prefix, obs_embeds], dim=1)

        B, K, _ = latent_prefix.shape
        _, S, _ = obs_embeds.shape
        prefix_mask = torch.ones(B, K, device=obs_embeds.device)
        obs_mask = torch.ones(B, S, device=obs_embeds.device)
        full_mask = torch.cat([prefix_mask, obs_mask], dim=1)

        return combined, full_mask


class CrossAttentionBridge(nn.Module):
    """
    Alternative receiver injection via cross-attention:
    the receiver's query tokens attend to the latent message tokens
    before generating the response.
    """

    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.scale = nn.Parameter(torch.zeros(1))  # start at 0 (no latent influence)

    def forward(
        self,
        query: torch.Tensor,    # [B, seq_len, hidden_size]
        latent: torch.Tensor,   # [B, K, hidden_size]
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns: enriched query [B, seq_len, hidden_size]
        """
        attn_out, _ = self.cross_attn(query, latent, latent, key_padding_mask=key_padding_mask)
        return self.norm(query + torch.tanh(self.scale) * attn_out)


# ─────────────────────────────────────────────────────────────────────────────
#  Continuous and VQ Latent Channels
# ─────────────────────────────────────────────────────────────────────────────

class ContinuousLatentChannel(nn.Module):
    """
    Mode 1: Pass raw continuous latent tokens from sender to receiver.
    No quantization — maximum information, higher bandwidth.
    Includes reconstruction loss to prevent mode collapse.
    """

    def __init__(self, hidden_size: int, latent_dim: int = 256, recon_weight: float = 0.5):
        super().__init__()
        self.encoder = LatentEncoder(hidden_size, latent_dim)
        self.decoder = LatentDecoder(latent_dim, hidden_size)
        self.injector = SoftPromptPrefixInjector(hidden_size)
        self.recon_weight = recon_weight

    def send(self, hidden_summary: torch.Tensor) -> torch.Tensor:
        """Encode hidden states to latent space. Returns [B, K, latent_dim]"""
        return self.encoder(hidden_summary)

    def receive(
        self,
        latent: torch.Tensor,           # [B, K, latent_dim]
        obs_embeds: torch.Tensor,       # [B, seq_len, hidden_size]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode and inject into receiver's input. Returns (combined_embeds, mask)"""
        decoded = self.decoder(latent)  # [B, K, hidden_size]
        return self.injector(obs_embeds, decoded)

    def forward(
        self,
        hidden_summary: torch.Tensor,
        obs_embeds: torch.Tensor,
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = self.send(hidden_summary)
        combined, mask = self.receive(latent, obs_embeds)
        # Reconstruction loss: encoder→latent→decoder should reconstruct original
        reconstructed = self.decoder(latent)
        recon_loss = F.mse_loss(reconstructed, hidden_summary) * self.recon_weight
        return combined, mask, recon_loss


class VQLatentChannel(nn.Module):
    """
    Mode 2: Vector-quantized latent communication.
    Sender quantizes to codebook entries → discrete token IDs transmitted.
    Receiver looks up codebook and injects decoded vectors.
    """

    def __init__(
        self,
        hidden_size: int,
        latent_dim: int = 256,
        codebook_size: int = 512,
        commitment_cost: float = 0.25,
        ema_decay: float = 0.99,
        use_gumbel_warmup: bool = True,
    ):
        super().__init__()
        self.encoder = LatentEncoder(hidden_size, latent_dim)
        self.vq = VectorQuantizer(
            codebook_size=codebook_size,
            embedding_dim=latent_dim,
            commitment_cost=commitment_cost,
            ema_decay=ema_decay,
            use_gumbel_warmup=use_gumbel_warmup,
        )
        self.decoder = LatentDecoder(latent_dim, hidden_size)
        self.injector = SoftPromptPrefixInjector(hidden_size)

    def send(
        self, hidden_summary: torch.Tensor, training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode + quantize. Returns:
            z_q: quantized latent [B, K, latent_dim]
            vq_loss: commitment + codebook loss
            indices: codebook IDs [B, K]
        """
        z = self.encoder(hidden_summary)
        z_q, vq_loss, indices = self.vq.quantize(z, training=training)
        return z_q, vq_loss, indices

    def receive(
        self,
        z_q: torch.Tensor,        # [B, K, latent_dim] (or indices if remote)
        obs_embeds: torch.Tensor, # [B, seq_len, hidden_size]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        decoded = self.decoder(z_q)
        return self.injector(obs_embeds, decoded)

    def receive_from_indices(
        self,
        indices: torch.LongTensor,  # [B, K]
        obs_embeds: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Look up codebook from discrete IDs (for simulation of bandwidth savings)."""
        z_q = self.vq.codebook(indices)  # [B, K, latent_dim]
        return self.receive(z_q, obs_embeds)

    def forward(
        self,
        hidden_summary: torch.Tensor,
        obs_embeds: torch.Tensor,
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_q, vq_loss, indices = self.send(hidden_summary, training=training)
        combined, mask = self.receive(z_q, obs_embeds)
        return combined, mask, vq_loss


# ─────────────────────────────────────────────────────────────────────────────
#  Unified LatentCommunicationModule
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CommModuleConfig:
    """Configuration for the latent communication module."""
    mode: Literal["continuous", "vq", "text"] = "continuous"
    hidden_size: int = 3072
    latent_dim: int = 256
    codebook_size: int = 512
    commitment_cost: float = 0.25
    ema_decay: float = 0.99
    use_gumbel_warmup: bool = True
    injection_mode: Literal["prefix", "cross_attention"] = "prefix"
    num_cross_attn_heads: int = 8


class LatentCommunicationModule(nn.Module):
    """
    Unified communication module supporting both continuous and VQ latent modes.
    Text mode falls through (no latent processing — baseline comparison).
    """

    def __init__(self, cfg: CommModuleConfig):
        super().__init__()
        self.cfg = cfg
        self.mode = cfg.mode

        if cfg.mode == "continuous":
            self.channel = ContinuousLatentChannel(cfg.hidden_size, cfg.latent_dim)
        elif cfg.mode == "vq":
            self.channel = VQLatentChannel(
                hidden_size=cfg.hidden_size,
                latent_dim=cfg.latent_dim,
                codebook_size=cfg.codebook_size,
                commitment_cost=cfg.commitment_cost,
                ema_decay=cfg.ema_decay,
                use_gumbel_warmup=cfg.use_gumbel_warmup,
            )
        else:
            self.channel = None  # text mode — no latent communication

        # Cross-attention bridge (optional alternative to prefix)
        if cfg.injection_mode == "cross_attention" and cfg.mode != "text":
            self.bridge = CrossAttentionBridge(
                cfg.hidden_size, num_heads=cfg.num_cross_attn_heads
            )
        else:
            self.bridge = None

    def forward(
        self,
        hidden_summary: Optional[torch.Tensor],
        obs_embeds: torch.Tensor,
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Process latent communication for one sender→receiver pair.

        Args:
            hidden_summary: Sender's summarized hidden states [B, K, hidden_size]
                            (None in text mode)
            obs_embeds: Receiver's observation embeddings [B, seq_len, hidden_size]
            training: Whether we are in training mode

        Returns:
            combined_embeds: Input to receiver backbone [B, *, hidden_size]
            attention_mask: Valid positions [B, *]
            info: Dict with loss, indices, mode details
        """
        info: Dict = {"mode": self.mode, "vq_loss": 0.0, "indices": None}

        if self.mode == "text" or hidden_summary is None or self.channel is None:
            # Text baseline: no latent injection
            B, S, _ = obs_embeds.shape
            mask = torch.ones(B, S, device=obs_embeds.device)
            return obs_embeds, mask, info

        combined, mask, loss = self.channel(
            hidden_summary, obs_embeds, training=training
        )
        # Keep tensor for training (gradient flow), store float for logging
        info["vq_loss"] = loss  # keep as tensor for backprop if training
        info["vq_loss_value"] = loss.item() if isinstance(loss, torch.Tensor) else float(loss)

        if hasattr(self.channel, "vq") and self.mode == "vq":
            # Retrieve last computed indices for logging
            pass  # indices logged inside VQLatentChannel.send()

        return combined, mask, info

    def get_codebook_ids(
        self, hidden_summary: torch.Tensor
    ) -> Optional[torch.LongTensor]:
        """Returns codebook indices for VQ mode (for audit logging)."""
        if self.mode != "vq":
            return None
        z = self.channel.encoder(hidden_summary)
        _, _, indices = self.channel.vq.quantize(z, training=False)
        return indices  # [B, K]
