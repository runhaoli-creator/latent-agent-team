"""
sft_bootstrap.py — Stage 1: Supervised Communication Bootstrapping.

Pipeline:
  1. Load teacher traces (text-based multi-agent trajectories)
  2. Build TeacherTraceDataset from Mind2Web/WebShop/AgentBench train splits
  3. Train latent encoder/decoder + role adapters with multi-task loss:
       - next_action_ce:       next-action cross-entropy loss
       - constraint_recon:     entity/constraint reconstruction loss
       - audit_decode_loss:    optional audit-text decoding loss
       - routing_supervision:  routing supervision from teacher recipients
       - bitrate_reg:          bitrate regularizer (penalize high K)
  4. All base backbone weights remain frozen; only LoRA + comm modules train.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup

from ..models.backbone import BackboneManager
from ..models.role_adapter import RoleAdapterManager
from ..models.latent_comm import LatentCommunicationModule
from ..models.router import AdaptiveBitrateScheduler, SparseRouter
from ..models.audit_decoder import AuditDecoder

logger = logging.getLogger(__name__)


# ── Teacher Trace Data Structures ────────────────────────────────────────────

@dataclass
class TraceStep:
    """One step in a teacher trace."""
    step_id: int
    episode_id: str
    sender_role: str
    observation: str
    outgoing_text_message: str
    next_action: str
    selected_evidence: str            # DOM/product/tool evidence
    verifier_label: str               # "pass" | "fail" | "uncertain"
    recipient_roles: List[str]        # teacher routing decisions
    benchmark: str                    # "mind2web" | "webshop" | "agentbench"
    task_instruction: str


class TeacherTraceDataset(Dataset):
    """
    Dataset of teacher traces for SFT bootstrapping.
    Loads from JSONL files: one record per trajectory step.
    """

    def __init__(
        self,
        trace_files: List[str],
        tokenizer: Any,
        max_seq_len: int = 2048,
        benchmarks: Optional[List[str]] = None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.benchmarks = benchmarks or ["mind2web", "webshop", "agentbench"]
        self.traces: List[TraceStep] = []
        self._load_traces(trace_files)

    def _load_traces(self, files: List[str]) -> None:
        for fpath in files:
            if not os.path.exists(fpath):
                logger.warning(f"Trace file not found: {fpath}")
                continue
            with open(fpath) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        if d.get("benchmark") not in self.benchmarks:
                            continue
                        step = TraceStep(
                            step_id=d.get("step_id", 0),
                            episode_id=d.get("episode_id", ""),
                            sender_role=d.get("sender_role", "planner"),
                            observation=d.get("observation", ""),
                            outgoing_text_message=d.get("outgoing_text_message", ""),
                            next_action=d.get("next_action", ""),
                            selected_evidence=d.get("selected_evidence", ""),
                            verifier_label=d.get("verifier_label", "uncertain"),
                            recipient_roles=d.get("recipient_roles", []),
                            benchmark=d.get("benchmark", "webshop"),
                            task_instruction=d.get("task_instruction", ""),
                        )
                        self.traces.append(step)
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"Skipping malformed trace line: {e}")
        logger.info(f"Loaded {len(self.traces)} teacher trace steps from {len(files)} files")

    def __len__(self) -> int:
        return len(self.traces)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        trace = self.traces[idx]

        # Build input sequence: role_prompt + observation + message
        input_text = (
            f"[ROLE]: {trace.sender_role}\n"
            f"[TASK]: {trace.task_instruction[:200]}\n"
            f"[OBS]: {trace.observation[:400]}\n"
            f"[MSG]: {trace.outgoing_text_message[:200]}\n"
            f"[ACTION]:"
        )

        # Target: next action
        target_text = trace.next_action[:200]

        # Encode inputs
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_len - 128,
            padding="max_length",
        )

        # Encode target action
        target_enc = self.tokenizer(
            target_text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding="max_length",
        )

        # Encode outgoing text message (for audit decoder target)
        msg_enc = self.tokenizer(
            trace.outgoing_text_message[:128],
            return_tensors="pt",
            truncation=True,
            max_length=64,
            padding="max_length",
        )

        # Encode selected evidence (for constraint reconstruction)
        evid_enc = self.tokenizer(
            trace.selected_evidence[:256],
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding="max_length",
        )

        # Teacher routing as binary vector
        agent_names = ["planner", "retriever", "browser", "verifier", "memory"]
        routing_target = torch.tensor(
            [1.0 if a in trace.recipient_roles else 0.0 for a in agent_names]
        )

        # Verifier label
        label_map = {"pass": 0, "fail": 1, "uncertain": 2}
        verifier_label = label_map.get(trace.verifier_label, 2)

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "target_ids": target_enc["input_ids"].squeeze(0),
            "target_mask": target_enc["attention_mask"].squeeze(0),
            "audit_target_ids": msg_enc["input_ids"].squeeze(0),
            "evidence_ids": evid_enc["input_ids"].squeeze(0),
            "routing_target": routing_target,
            "verifier_label": torch.tensor(verifier_label, dtype=torch.long),
            "sender_role": trace.sender_role,
            "benchmark": trace.benchmark,
        }

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
        keys_to_stack = [
            "input_ids", "attention_mask", "target_ids", "target_mask",
            "audit_target_ids", "evidence_ids", "routing_target", "verifier_label",
        ]
        collated: Dict[str, Any] = {}
        for k in keys_to_stack:
            collated[k] = torch.stack([b[k] for b in batch], dim=0)
        collated["sender_roles"] = [b["sender_role"] for b in batch]
        collated["benchmarks"] = [b["benchmark"] for b in batch]
        return collated


# ── Multi-Task Loss ───────────────────────────────────────────────────────────

@dataclass
class LossWeights:
    next_action_ce: float = 1.0
    constraint_recon: float = 0.3
    audit_decode: float = 0.1
    routing_supervision: float = 0.2
    bitrate_reg: float = 0.01
    vq_commitment: float = 1.0


class MultiTaskLoss(nn.Module):
    """
    Combined multi-task loss for SFT bootstrapping.

    L = w1 * L_action + w2 * L_constraint + w3 * L_audit
        + w4 * L_routing + w5 * L_bitrate + w6 * L_vq
    """

    def __init__(self, weights: Optional[LossWeights] = None):
        super().__init__()
        self.w = weights or LossWeights()

    def forward(
        self,
        # Next-action prediction
        action_logits: torch.Tensor,   # [B, T, vocab_size]
        action_targets: torch.Tensor,  # [B, T]
        # Constraint/evidence reconstruction
        evid_logits: Optional[torch.Tensor] = None,
        evid_targets: Optional[torch.Tensor] = None,
        # Audit text decoding
        audit_logits: Optional[torch.Tensor] = None,
        audit_targets: Optional[torch.Tensor] = None,
        # Routing supervision
        routing_probs: Optional[torch.Tensor] = None,
        routing_targets: Optional[torch.Tensor] = None,
        # Bitrate regularization
        k_probs: Optional[torch.Tensor] = None,
        # VQ commitment loss
        vq_loss: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        device = action_logits.device
        loss_dict: Dict[str, float] = {}
        total = torch.tensor(0.0, device=device)

        # 1. Next-action cross-entropy
        B, T, V = action_logits.shape
        action_loss = F.cross_entropy(
            action_logits[:, :-1].reshape(-1, V),
            action_targets[:, 1:].reshape(-1),
            ignore_index=0,
        )
        total = total + self.w.next_action_ce * action_loss
        loss_dict["action_ce"] = float(action_loss.item())

        # 2. Constraint/evidence reconstruction
        if evid_logits is not None and evid_targets is not None:
            B2, T2, V2 = evid_logits.shape
            evid_loss = F.cross_entropy(
                evid_logits[:, :-1].reshape(-1, V2),
                evid_targets[:, 1:].reshape(-1),
                ignore_index=0,
            )
            total = total + self.w.constraint_recon * evid_loss
            loss_dict["constraint_recon"] = float(evid_loss.item())

        # 3. Audit text decoding loss
        if audit_logits is not None and audit_targets is not None:
            B3, T3, V3 = audit_logits.shape
            audit_loss = F.cross_entropy(
                audit_logits[:, :-1].reshape(-1, V3),
                audit_targets[:, 1:].reshape(-1),
                ignore_index=0,
            )
            total = total + self.w.audit_decode * audit_loss
            loss_dict["audit_decode"] = float(audit_loss.item())

        # 4. Routing supervision
        if routing_probs is not None and routing_targets is not None:
            routing_loss = F.binary_cross_entropy(
                routing_probs.clamp(1e-7, 1 - 1e-7),
                routing_targets.float(),
            )
            total = total + self.w.routing_supervision * routing_loss
            loss_dict["routing"] = float(routing_loss.item())

        # 5. Bitrate regularization
        if k_probs is not None:
            k_vals = torch.tensor(
                [4., 8., 16., 32., 64.], device=device
            )
            expected_k = (k_probs * k_vals).sum(dim=-1).mean()
            bitrate_loss = expected_k / 64.0  # normalize to [0,1]
            total = total + self.w.bitrate_reg * bitrate_loss
            loss_dict["bitrate_reg"] = float(bitrate_loss.item())

        # 6. VQ commitment loss
        if vq_loss is not None and isinstance(vq_loss, torch.Tensor):
            total = total + self.w.vq_commitment * vq_loss
            loss_dict["vq_commitment"] = float(vq_loss.item())

        loss_dict["total"] = float(total.item())
        return total, loss_dict


# ── SFT Bootstrapper ─────────────────────────────────────────────────────────

@dataclass
class SFTConfig:
    # Data
    trace_files: List[str] = field(default_factory=list)
    benchmarks: List[str] = field(default_factory=lambda: ["mind2web", "webshop", "agentbench"])

    # Training
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    max_seq_len: int = 2048

    # Communication
    comm_mode: str = "continuous"   # "continuous" | "vq" | "text"
    k_initial: int = 8              # starting K for latent tokens

    # Logging
    output_dir: str = "outputs/sft"
    log_every: int = 10
    save_every: int = 500
    use_wandb: bool = False
    project_name: str = "latent_agent_team"
    run_name: str = "sft_bootstrap"

    # Loss weights
    loss_weights: LossWeights = field(default_factory=LossWeights)


class SFTBootstrapper:
    """
    Stage 1 training: supervised bootstrapping of latent communication modules.

    Trains on teacher traces where text messages are used to provide
    supervision signal for the latent encoder/decoder.
    """

    def __init__(
        self,
        cfg: SFTConfig,
        backbone: BackboneManager,
        role_manager: RoleAdapterManager,
        comm_module: LatentCommunicationModule,
        bitrate_scheduler: AdaptiveBitrateScheduler,
        sparse_router: SparseRouter,
        audit_decoder: AuditDecoder,
    ):
        self.cfg = cfg
        self.backbone = backbone
        self.role_manager = role_manager
        self.comm_module = comm_module
        self.bitrate_scheduler = bitrate_scheduler
        self.sparse_router = sparse_router
        self.audit_decoder = audit_decoder
        self.loss_fn = MultiTaskLoss(cfg.loss_weights)

        os.makedirs(cfg.output_dir, exist_ok=True)

    def _get_trainable_params(self) -> List[nn.Parameter]:
        """Collect all trainable parameters (LoRA adapters + comm modules)."""
        params = []
        # LoRA adapter parameters
        for name, p in self.backbone.model.named_parameters():
            if p.requires_grad:
                params.append(p)
        # Role adapter summarizers
        for _, p in self.role_manager.get_all_summarizer_params():
            params.append(p)
        # Communication module
        if self.comm_module.channel is not None:
            params.extend(self.comm_module.channel.parameters())
        # Router + bitrate scheduler
        params.extend(self.bitrate_scheduler.parameters())
        params.extend(self.sparse_router.parameters())
        # Audit decoder
        params.extend(self.audit_decoder.parameters())
        return params

    def build_dataloader(self, split: str = "train") -> DataLoader:
        dataset = TeacherTraceDataset(
            trace_files=self.cfg.trace_files,
            tokenizer=self.backbone.tokenizer,
            max_seq_len=self.cfg.max_seq_len,
            benchmarks=self.cfg.benchmarks,
        )
        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=(split == "train"),
            num_workers=4,
            pin_memory=True,
            collate_fn=TeacherTraceDataset.collate_fn,
        )

    def _forward_step(
        self,
        batch: Dict[str, Any],
        device: torch.device,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Run one training forward pass."""
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        target_ids = batch["target_ids"].to(device)
        audit_target_ids = batch["audit_target_ids"].to(device)
        routing_targets = batch["routing_target"].to(device)

        # Clamp audit target IDs to audit decoder's vocab range
        # (audit decoder has its own embedding which may differ from backbone)
        audit_vocab_size = self.audit_decoder.text_decoder.vocab_size
        audit_target_ids = audit_target_ids.clamp(0, audit_vocab_size - 1)
        # Clamp backbone input/target IDs to model's actual vocab range
        model_vocab_size = self.backbone.model.config.vocab_size
        input_ids = input_ids.clamp(0, model_vocab_size - 1)
        target_ids = target_ids.clamp(0, model_vocab_size - 1)

        # Determine K for this batch from average uncertainty features
        B = input_ids.size(0)
        k = self.cfg.k_initial  # Fixed K during SFT; adaptive during DPO

        # ── Step 1: Get backbone hidden states (as "sender") ─────────────
        sender_role = batch["sender_roles"][0]
        self.backbone.set_active_adapter(sender_role)

        hidden_states = self.backbone.get_hidden_states(input_ids, attention_mask)

        # ── Step 2: Get role adapter summarization ────────────────────────
        role_adapter = self.role_manager.get_adapter(sender_role)
        latent_summary = role_adapter.summarize_hidden_states(
            hidden_states, attention_mask, k=k
        )  # [B, K, hidden_size]

        # ── Step 3: Communication forward ────────────────────────────────
        obs_embeds = self.backbone.model.get_input_embeddings()(input_ids)
        combined_embeds, combined_mask, comm_info = self.comm_module(
            hidden_summary=latent_summary,
            obs_embeds=obs_embeds,
            training=True,
        )
        vq_loss_val = comm_info.get("vq_loss", 0.0)
        vq_loss = torch.tensor(vq_loss_val, device=device) if isinstance(vq_loss_val, float) else vq_loss_val

        # ── Step 4: Action prediction (receiver forward) ──────────────────
        # Use combined_embeds as input to backbone
        action_out = self.backbone.model(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
        )
        action_logits = action_out.logits  # [B, T, vocab_size]

        # Align with target_ids by trimming or padding
        T_action = action_logits.size(1)
        T_target = target_ids.size(1)
        if T_action > T_target:
            action_logits = action_logits[:, :T_target, :]
        elif T_action < T_target:
            target_ids = target_ids[:, :T_action]

        # ── Step 5: Audit decoder ─────────────────────────────────────────
        audit_logits, audit_loss = self.audit_decoder(latent_summary, audit_target_ids)

        # ── Step 6: Routing prediction ────────────────────────────────────
        sender_idx = ["planner", "retriever", "browser", "verifier", "memory"].index(sender_role)
        routing_probs, _ = self.sparse_router(
            sender_idx=sender_idx,
            latent_tokens=latent_summary,
            hard=False,
        )

        # ── Step 7: Bitrate prediction ────────────────────────────────────
        feat = torch.zeros(B, self.bitrate_scheduler.INPUT_DIM, device=device)
        k_probs, _ = self.bitrate_scheduler(feat, hard=False)

        # ── Step 8: Compute multi-task loss ───────────────────────────────
        T_audit = audit_logits.size(1)
        audit_target = audit_target_ids[:, :T_audit]

        total_loss, loss_dict = self.loss_fn(
            action_logits=action_logits,
            action_targets=target_ids,
            audit_logits=audit_logits,
            audit_targets=audit_target,
            routing_probs=routing_probs,
            routing_targets=routing_targets,
            k_probs=k_probs,
            vq_loss=vq_loss if isinstance(vq_loss, torch.Tensor) else None,
        )

        return total_loss, loss_dict

    def train(self, device: Optional[torch.device] = None) -> None:
        """Run Stage 1 SFT training."""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Starting SFT bootstrapping on {device}")
        logger.info(f"Config: {self.cfg}")

        # ── Optimizer ────────────────────────────────────────────────────
        trainable_params = self._get_trainable_params()
        trainable_count = sum(p.numel() for p in trainable_params)
        logger.info(f"Trainable parameters: {trainable_count / 1e6:.2f}M")

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )

        # ── Dataloader ───────────────────────────────────────────────────
        dataloader = self.build_dataloader("train")
        total_steps = len(dataloader) * self.cfg.num_epochs

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.cfg.warmup_steps,
            num_training_steps=total_steps // self.cfg.gradient_accumulation_steps,
        )

        # ── Optional WandB ───────────────────────────────────────────────
        if self.cfg.use_wandb:
            try:
                import wandb
                wandb.init(project=self.cfg.project_name, name=self.cfg.run_name)
            except ImportError:
                logger.warning("wandb not available, skipping logging")

        # ── Training loop ────────────────────────────────────────────────
        global_step = 0
        best_loss = float("inf")

        for epoch in range(self.cfg.num_epochs):
            epoch_loss = 0.0
            epoch_start = time.time()

            for step, batch in enumerate(dataloader):
                loss, loss_dict = self._forward_step(batch, device)
                loss = loss / self.cfg.gradient_accumulation_steps
                loss.backward()

                epoch_loss += loss.item()

                if (step + 1) % self.cfg.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        trainable_params, self.cfg.max_grad_norm
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if global_step % self.cfg.log_every == 0:
                        lr = scheduler.get_last_lr()[0]
                        log_msg = (
                            f"Epoch {epoch+1}/{self.cfg.num_epochs} | "
                            f"Step {global_step} | "
                            f"Loss: {loss_dict.get('total', 0):.4f} | "
                            f"Action CE: {loss_dict.get('action_ce', 0):.4f} | "
                            f"LR: {lr:.2e}"
                        )
                        logger.info(log_msg)

                        if self.cfg.use_wandb:
                            try:
                                import wandb
                                wandb.log({"step": global_step, **loss_dict, "lr": lr})
                            except Exception:
                                pass

                    if global_step % self.cfg.save_every == 0:
                        ckpt_dir = os.path.join(self.cfg.output_dir, f"checkpoint_{global_step}")
                        self.save_checkpoint(ckpt_dir)
                        logger.info(f"Saved checkpoint to {ckpt_dir}")

            epoch_time = time.time() - epoch_start
            avg_loss = epoch_loss / len(dataloader)
            logger.info(
                f"Epoch {epoch+1} done. Avg loss: {avg_loss:.4f}. "
                f"Time: {epoch_time:.1f}s"
            )

            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_checkpoint(os.path.join(self.cfg.output_dir, "best"))

        # ── Final save ───────────────────────────────────────────────────
        self.save_checkpoint(os.path.join(self.cfg.output_dir, "final"))
        logger.info(f"SFT training complete. Best loss: {best_loss:.4f}")

    def save_checkpoint(self, save_dir: str) -> None:
        """Save all trainable modules."""
        os.makedirs(save_dir, exist_ok=True)

        # Save LoRA adapters
        self.backbone.save_adapters(os.path.join(save_dir, "lora_adapters"))

        # Save communication module
        if self.comm_module.channel is not None:
            torch.save(
                self.comm_module.state_dict(),
                os.path.join(save_dir, "comm_module.pt"),
            )

        # Save router + bitrate scheduler
        torch.save(
            self.bitrate_scheduler.state_dict(),
            os.path.join(save_dir, "bitrate_scheduler.pt"),
        )
        torch.save(
            self.sparse_router.state_dict(),
            os.path.join(save_dir, "sparse_router.pt"),
        )

        # Save audit decoder
        torch.save(
            self.audit_decoder.state_dict(),
            os.path.join(save_dir, "audit_decoder.pt"),
        )

        # Save role summarizers
        for role_name, adapter in self.role_manager.adapters.items():
            torch.save(
                adapter.summarizer.state_dict(),
                os.path.join(save_dir, f"summarizer_{role_name}.pt"),
            )

        logger.info(f"Checkpoint saved to {save_dir}")


def main():
    """CLI entry point for SFT training."""
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(description="Stage 1: SFT Bootstrapping")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--backbone", type=str, default="phi3_mini")
    parser.add_argument("--comm_mode", type=str, default="continuous")
    parser.add_argument("--output_dir", type=str, default="outputs/sft")
    args = parser.parse_args()

    cfg_dict = OmegaConf.load(args.config)
    sft_cfg = SFTConfig(
        output_dir=args.output_dir,
        comm_mode=args.comm_mode,
        **{k: v for k, v in cfg_dict.items() if k in SFTConfig.__dataclass_fields__},
    )

    # Import and build the full pipeline
    from ..models.backbone import BackboneConfig, BackboneManager
    from ..models.role_adapter import RoleAdapterManager
    from ..models.latent_comm import LatentCommunicationModule, CommModuleConfig
    from ..models.router import AdaptiveBitrateScheduler, SparseRouter
    from ..models.audit_decoder import AuditDecoder

    backbone_cfg = BackboneConfig(backbone_name=args.backbone)
    backbone = BackboneManager(backbone_cfg)

    role_manager = RoleAdapterManager(backbone.hidden_size)
    for role in role_manager.roles:
        backbone.add_adapter(role)

    comm_cfg = CommModuleConfig(mode=sft_cfg.comm_mode, hidden_size=backbone.hidden_size)
    comm_module = LatentCommunicationModule(comm_cfg)

    bitrate_scheduler = AdaptiveBitrateScheduler()
    sparse_router = SparseRouter(hidden_size=backbone.hidden_size)
    audit_decoder = AuditDecoder(
        hidden_size=backbone.hidden_size,
        vocab_size=backbone.model.config.vocab_size,
    )

    trainer = SFTBootstrapper(
        cfg=sft_cfg,
        backbone=backbone,
        role_manager=role_manager,
        comm_module=comm_module,
        bitrate_scheduler=bitrate_scheduler,
        sparse_router=sparse_router,
        audit_decoder=audit_decoder,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer.train(device=device)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
