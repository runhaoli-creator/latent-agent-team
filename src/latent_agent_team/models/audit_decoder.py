"""
audit_decoder.py — Decodes latent tokens back to human-readable text for
auditability and interpretability.

The audit channel provides:
  1. A lightweight text decoder that maps latent message embeddings → natural language
  2. A disagreement detector that flags when decoded intent ≠ receiver's action
  3. Failure annotation: marks when audit messages correlate with task failures
  4. Structured audit records for JSONL logging

This is critical for the paper's central claim: latent communication should be
budgeted AND auditable. The audit decoder provides transparency without forcing
agents to communicate purely in text.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class AuditRecord:
    """
    Structured record for one latent communication event.
    Saved to JSONL for post-hoc analysis.
    """
    step: int
    episode_id: str
    sender_role: str
    recipient_roles: List[str]
    k_selected: int
    mode: str  # "continuous" | "vq" | "text"
    codebook_ids: Optional[List[int]]  # VQ mode only
    decoded_audit_text: str
    receiver_action: str
    reward: float
    is_failure: bool
    disagreement_score: float  # 0-1: how much audit text contradicts action
    task_context: str
    latent_tokens_hex: Optional[str] = None  # optional: first few bytes for debug


class AuditTextDecoder(nn.Module):
    """
    Lightweight seq2seq head that decodes latent tokens into readable text.

    Architecture:
        latent_tokens [B, K, hidden_size]
            → pool + project to decoder_dim
            → 2-layer Transformer decoder with learned text queries
            → linear to vocabulary
    """

    def __init__(
        self,
        hidden_size: int = 3072,
        vocab_size: int = 32000,
        decoder_dim: int = 256,
        num_decoder_layers: int = 2,
        num_heads: int = 4,
        max_decode_len: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.decoder_dim = decoder_dim
        self.max_decode_len = max_decode_len

        # Project latent to decoder dim
        self.latent_proj = nn.Linear(hidden_size, decoder_dim)
        self.latent_norm = nn.LayerNorm(decoder_dim)

        # Learned positional text queries for decoding
        self.text_queries = nn.Embedding(max_decode_len, decoder_dim)

        # Transformer decoder (text_queries attend to latent context)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=decoder_dim,
            nhead=num_heads,
            dim_feedforward=decoder_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Linear head to vocabulary
        self.vocab_proj = nn.Linear(decoder_dim, vocab_size)

        # Token embedding for teacher-forced decoding
        self.token_embed = nn.Embedding(vocab_size, decoder_dim)
        nn.init.normal_(self.token_embed.weight, std=0.02)

    def forward(
        self,
        latent_tokens: torch.Tensor,    # [B, K, hidden_size]
        target_ids: Optional[torch.LongTensor] = None,  # [B, decode_len] for training
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            latent_tokens: Sender's latent message [B, K, hidden_size]
            target_ids: Teacher-forced target tokens [B, T] (training only)

        Returns:
            logits: Token logits [B, T or max_decode_len, vocab_size]
            loss:   Cross-entropy loss (0 if no target_ids)
        """
        B = latent_tokens.size(0)
        device = latent_tokens.device

        # Encode latent context
        memory = self.latent_norm(self.latent_proj(latent_tokens))  # [B, K, decoder_dim]

        if target_ids is not None:
            # Teacher-forced training
            T = target_ids.size(1)
            tgt_embed = self.token_embed(target_ids)  # [B, T, decoder_dim]

            # Causal mask to prevent attending to future tokens
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(T, device=device)

            out = self.decoder(tgt_embed, memory, tgt_mask=tgt_mask)  # [B, T, decoder_dim]
            logits = self.vocab_proj(out)  # [B, T, vocab_size]

            # Compute cross-entropy loss (shift by 1 for next-token prediction)
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, self.vocab_size),
                target_ids[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        else:
            # Inference: use learned positional queries
            pos_ids = torch.arange(self.max_decode_len, device=device).unsqueeze(0).expand(B, -1)
            queries = self.text_queries(pos_ids)  # [B, max_len, decoder_dim]
            out = self.decoder(queries, memory)   # [B, max_len, decoder_dim]
            logits = self.vocab_proj(out)          # [B, max_len, vocab_size]
            loss = torch.tensor(0.0, device=device)

        return logits, loss

    @torch.no_grad()
    def decode_greedy(
        self,
        latent_tokens: torch.Tensor,  # [B, K, hidden_size]
        tokenizer: Any,
        max_len: Optional[int] = None,
    ) -> List[str]:
        """
        Greedy decode latent tokens to text for audit logging.

        Returns list of decoded strings, one per batch element.
        """
        max_len = max_len or self.max_decode_len
        logits, _ = self.forward(latent_tokens, target_ids=None)
        token_ids = logits.argmax(dim=-1)  # [B, max_len]

        decoded_texts = []
        for i in range(token_ids.size(0)):
            ids = token_ids[i].tolist()
            # Stop at eos if present
            if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id in ids:
                ids = ids[: ids.index(tokenizer.eos_token_id)]
            text = tokenizer.decode(ids, skip_special_tokens=True)
            decoded_texts.append(text.strip())

        return decoded_texts


class DisagreementDetector(nn.Module):
    """
    Detects disagreement between the decoded audit message and the receiver's
    chosen action. A high disagreement score indicates the latent channel
    transmitted content that was ignored or contradicted by the receiver.

    Used for:
      1. Failure analysis in the ablation notebook
      2. A secondary supervision signal during training
    """

    def __init__(self, embed_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim

        # Simple cosine-similarity-based detector on embeddings
        # (In practice, embeddings come from a frozen sentence encoder or
        #  directly from latent tokens vs action token embeddings)
        self.audit_proj = nn.Linear(embed_dim, embed_dim)
        self.action_proj = nn.Linear(embed_dim, embed_dim)
        self.score_head = nn.Sequential(
            nn.Linear(embed_dim * 2 + 1, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        audit_embed: torch.Tensor,   # [B, embed_dim] mean of audit token embeddings
        action_embed: torch.Tensor,  # [B, embed_dim] mean of action token embeddings
    ) -> torch.Tensor:
        """
        Returns disagreement score in [0, 1].
        0 = perfectly aligned, 1 = completely misaligned.
        """
        a = F.normalize(self.audit_proj(audit_embed), dim=-1)
        b = F.normalize(self.action_proj(action_embed), dim=-1)

        cos_sim = (a * b).sum(dim=-1, keepdim=True)  # [B, 1]
        combined = torch.cat([a, b, cos_sim], dim=-1)
        score = self.score_head(combined).squeeze(-1)  # [B]
        return score


class AuditDecoder(nn.Module):
    """
    Full audit pipeline combining:
    1. AuditTextDecoder — latent tokens → natural language
    2. DisagreementDetector — audit text vs receiver action alignment
    3. record_audit() — structured JSONL record creation
    """

    def __init__(
        self,
        hidden_size: int = 3072,
        vocab_size: int = 32000,
        decoder_dim: int = 256,
        max_decode_len: int = 32,
    ):
        super().__init__()
        self.text_decoder = AuditTextDecoder(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            decoder_dim=decoder_dim,
            max_decode_len=max_decode_len,
        )
        self.disagreement_detector = DisagreementDetector(embed_dim=decoder_dim)

    def forward(
        self,
        latent_tokens: torch.Tensor,
        target_ids: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Training forward: decode latent to text and compute loss.

        Returns:
            logits: [B, T, vocab_size]
            audit_loss: scalar cross-entropy loss
        """
        return self.text_decoder(latent_tokens, target_ids)

    @torch.no_grad()
    def generate_audit_text(
        self,
        latent_tokens: torch.Tensor,
        tokenizer: Any,
        max_len: int = 32,
    ) -> List[str]:
        """Generate human-readable audit descriptions of latent messages."""
        return self.text_decoder.decode_greedy(latent_tokens, tokenizer, max_len)

    def compute_disagreement(
        self,
        latent_tokens: torch.Tensor,  # [B, K, hidden_size]
        action_tokens: torch.Tensor,  # [B, T, hidden_size]
    ) -> torch.Tensor:
        """
        Compute disagreement between latent message intent and receiver's action.
        """
        audit_embed = latent_tokens.mean(dim=1)   # [B, hidden_size]
        action_embed = action_tokens.mean(dim=1)  # [B, hidden_size]

        # Project to detector dimension
        H = self.text_decoder.decoder_dim

        # Use first columns of projection weights for embedding alignment
        # (a simpler approach: mean-pool then linear)
        audit_proj = self.text_decoder.latent_proj
        B = latent_tokens.size(0)

        a_emb = audit_proj(audit_embed)    # [B, decoder_dim]
        b_emb = audit_proj(action_embed)   # [B, decoder_dim]

        score = self.disagreement_detector(a_emb, b_emb)
        return score

    def record_audit(
        self,
        step: int,
        episode_id: str,
        sender_role: str,
        recipient_roles: List[str],
        k_selected: int,
        mode: str,
        decoded_audit_text: str,
        receiver_action: str,
        reward: float,
        is_failure: bool,
        disagreement_score: float,
        task_context: str,
        codebook_ids: Optional[List[int]] = None,
    ) -> AuditRecord:
        """Create a structured audit record for JSONL logging."""
        return AuditRecord(
            step=step,
            episode_id=episode_id,
            sender_role=sender_role,
            recipient_roles=recipient_roles,
            k_selected=k_selected,
            mode=mode,
            codebook_ids=codebook_ids,
            decoded_audit_text=decoded_audit_text,
            receiver_action=receiver_action,
            reward=reward,
            is_failure=is_failure,
            disagreement_score=disagreement_score,
            task_context=task_context,
        )

    def audit_text_loss(
        self,
        latent_tokens: torch.Tensor,
        target_text_ids: torch.LongTensor,
        weight: float = 0.1,
    ) -> torch.Tensor:
        """
        Auxiliary loss: force audit decoder to reproduce teacher text messages
        from the latent tokens. Prevents latent channel from collapsing.
        """
        _, loss = self.text_decoder(latent_tokens, target_text_ids)
        return weight * loss
