"""
dpo_rollout.py — Stage 2: Online DPO preference optimization.

Strategy:
  1. Run interactive rollouts on training environments
  2. Sample multiple trajectories at different communication budgets
  3. Derive preference pairs: (success+low-cost) > (failure) or (success+low-cost) > (success+high-cost)
  4. Apply DPO (Direct Preference Optimization) to optimize success-minus-bandwidth objective
     - DPO is much simpler and lighter than RLHF (no reward model needed)
     - Directly optimizes preferred behaviors through contrastive log-likelihood

DPO loss:
  L_DPO = -E[log σ(β * (log π(y_w|x) - log π_ref(y_w|x))
                   - β * (log π(y_l|x) - log π_ref(y_l|x)))]

  Where y_w = preferred trajectory (success + low bandwidth)
        y_l = dispreferred trajectory (failure or high bandwidth)
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ..models.backbone import BackboneManager
from ..models.role_adapter import RoleAdapterManager
from ..models.latent_comm import LatentCommunicationModule
from ..models.router import AdaptiveBitrateScheduler, SparseRouter
from ..models.audit_decoder import AuditDecoder

logger = logging.getLogger(__name__)


@dataclass
class PreferencePair:
    """
    One preference pair for DPO training.
    y_w = preferred (success + low cost), y_l = dispreferred (failure or high cost).
    """
    task_instruction: str
    episode_id: str
    benchmark: str

    # Preferred trajectory
    y_w_tokens: List[int]       # token IDs of the winning trajectory actions
    y_w_k_used: int             # average K (latent tokens) used
    y_w_reward: float
    y_w_success: bool

    # Dispreferred trajectory
    y_l_tokens: List[int]
    y_l_k_used: int
    y_l_reward: float
    y_l_success: bool

    # Shared context
    context_tokens: List[int]   # observation/task tokens


class PreferenceDataset(Dataset):
    """Loads DPO preference pairs from JSONL files."""

    def __init__(
        self,
        preference_files: List[str],
        tokenizer: Any,
        max_seq_len: int = 1024,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pairs: List[PreferencePair] = []
        self._load(preference_files)

    def _load(self, files: List[str]) -> None:
        for fpath in files:
            if not os.path.exists(fpath):
                logger.warning(f"Preference file not found: {fpath}")
                continue
            with open(fpath) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        pair = PreferencePair(
                            task_instruction=d["task_instruction"],
                            episode_id=d["episode_id"],
                            benchmark=d.get("benchmark", "webshop"),
                            y_w_tokens=d["y_w_tokens"],
                            y_w_k_used=d.get("y_w_k_used", 8),
                            y_w_reward=d.get("y_w_reward", 1.0),
                            y_w_success=d.get("y_w_success", True),
                            y_l_tokens=d["y_l_tokens"],
                            y_l_k_used=d.get("y_l_k_used", 32),
                            y_l_reward=d.get("y_l_reward", 0.0),
                            y_l_success=d.get("y_l_success", False),
                            context_tokens=d["context_tokens"],
                        )
                        self.pairs.append(pair)
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"Skipping malformed pair: {e}")

        logger.info(f"Loaded {len(self.pairs)} preference pairs")

    def __len__(self) -> int:
        return len(self.pairs)

    def _pad_truncate(self, tokens: List[int], max_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        ids = torch.tensor(tokens, dtype=torch.long)
        mask = torch.ones_like(ids)
        pad_len = max_len - len(ids)
        if pad_len > 0:
            ids = F.pad(ids, (0, pad_len))
            mask = F.pad(mask, (0, pad_len))
        return ids, mask

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pair = self.pairs[idx]
        ctx_max = self.max_seq_len // 2
        resp_max = self.max_seq_len // 2

        ctx_ids, ctx_mask = self._pad_truncate(pair.context_tokens, ctx_max)
        yw_ids, yw_mask = self._pad_truncate(pair.y_w_tokens, resp_max)
        yl_ids, yl_mask = self._pad_truncate(pair.y_l_tokens, resp_max)

        return {
            "context_ids": ctx_ids,
            "context_mask": ctx_mask,
            "yw_ids": yw_ids,
            "yw_mask": yw_mask,
            "yl_ids": yl_ids,
            "yl_mask": yl_mask,
            "yw_k": torch.tensor(pair.y_w_k_used, dtype=torch.float),
            "yl_k": torch.tensor(pair.y_l_k_used, dtype=torch.float),
            "yw_reward": torch.tensor(pair.y_w_reward, dtype=torch.float),
            "yl_reward": torch.tensor(pair.y_l_reward, dtype=torch.float),
        }

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
        keys = ["context_ids", "context_mask", "yw_ids", "yw_mask",
                "yl_ids", "yl_mask", "yw_k", "yl_k", "yw_reward", "yl_reward"]
        return {k: torch.stack([b[k] for b in batch]) for k in keys}


# ── DPO Loss ─────────────────────────────────────────────────────────────────

def dpo_loss(
    policy_yw_logps: torch.Tensor,   # [B] log probs of preferred under policy
    policy_yl_logps: torch.Tensor,   # [B] log probs of dispreferred under policy
    ref_yw_logps: torch.Tensor,      # [B] log probs of preferred under reference
    ref_yl_logps: torch.Tensor,      # [B] log probs of dispreferred under reference
    beta: float = 0.1,
    bandwidth_penalty: Optional[torch.Tensor] = None,  # [B] per-pair bandwidth cost
    bandwidth_weight: float = 0.01,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    DPO loss with optional bandwidth penalty term.

    L_DPO = -E[log σ(β * (log π/π_ref(y_w|x) - log π/π_ref(y_l|x)))]
    L_bandwidth = weight * E[K_w]  (penalize high bandwidth even for wins)
    """
    # Log ratio differences
    yw_ratio = policy_yw_logps - ref_yw_logps   # [B]
    yl_ratio = policy_yl_logps - ref_yl_logps   # [B]

    # DPO objective
    logits = beta * (yw_ratio - yl_ratio)
    dpo = -F.logsigmoid(logits).mean()

    info: Dict[str, float] = {
        "dpo_loss": float(dpo.item()),
        "reward_margin": float((yw_ratio - yl_ratio).mean().item()),
        "preferred_reward": float(yw_ratio.mean().item()),
        "dispreferred_reward": float(yl_ratio.mean().item()),
    }

    total = dpo

    # Optional bandwidth penalty
    if bandwidth_penalty is not None:
        bw_loss = bandwidth_weight * bandwidth_penalty.mean()
        total = total + bw_loss
        info["bandwidth_penalty"] = float(bw_loss.item())

    info["total"] = float(total.item())
    return total, info


# ── Rollout Helper ────────────────────────────────────────────────────────────

class RolloutCollector:
    """
    Runs interactive rollouts across environments to collect preference pairs.

    For each task:
      1. Sample trajectory_budget trajectories at different K values
      2. Label with success/reward from environment
      3. Construct preference pairs: (success+low_K) > (fail or high_K)
    """

    def __init__(
        self,
        trajectory_budget: int = 4,
        k_choices: List[int] = None,
    ):
        self.trajectory_budget = trajectory_budget
        self.k_choices = k_choices or [4, 8, 32, 64]

    def collect_from_webshop(
        self,
        env: Any,
        team: Any,
        task: Dict[str, Any],
        device: torch.device,
    ) -> List[PreferencePair]:
        """
        Collect preference pairs from WebShop environment.
        env should expose: reset(), step(action) → (obs, reward, done, info)
        """
        pairs = []
        trajectories = []

        for k_val in self.k_choices[:self.trajectory_budget]:
            obs = env.reset()
            traj_tokens = []
            total_reward = 0.0
            done = False
            step_count = 0

            while not done and step_count < 20:
                # Run team step at this K budget
                action, hidden_states = team.step(obs, k_budget=k_val, device=device)
                action_str = str(action.get("action", ""))

                # Encode action for token tracking
                token_ids = team.backbone.tokenizer.encode(action_str)
                traj_tokens.extend(token_ids[:32])  # cap per step

                obs, reward, done, info = env.step(action_str)
                total_reward += float(reward)
                step_count += 1

            success = total_reward >= 0.5
            trajectories.append({
                "tokens": traj_tokens,
                "k_used": k_val,
                "reward": total_reward,
                "success": success,
            })

        # Construct preference pairs
        ctx_tokens = team.backbone.tokenizer.encode(
            task.get("instruction", "")[:200]
        )

        success_trajs = [t for t in trajectories if t["success"]]
        fail_trajs = [t for t in trajectories if not t["success"]]

        if success_trajs and fail_trajs:
            # Best success vs worst failure
            best = min(success_trajs, key=lambda t: t["k_used"])
            worst = max(fail_trajs, key=lambda t: t["k_used"])
            pairs.append(PreferencePair(
                task_instruction=task.get("instruction", ""),
                episode_id=task.get("episode_id", f"ep_{time.time_ns()}"),
                benchmark="webshop",
                y_w_tokens=best["tokens"],
                y_w_k_used=best["k_used"],
                y_w_reward=best["reward"],
                y_w_success=True,
                y_l_tokens=worst["tokens"],
                y_l_k_used=worst["k_used"],
                y_l_reward=worst["reward"],
                y_l_success=False,
                context_tokens=ctx_tokens,
            ))
        elif len(success_trajs) >= 2:
            # Both succeed: prefer lower bandwidth
            sorted_by_k = sorted(success_trajs, key=lambda t: t["k_used"])
            low_k, high_k = sorted_by_k[0], sorted_by_k[-1]
            if low_k["k_used"] < high_k["k_used"]:
                pairs.append(PreferencePair(
                    task_instruction=task.get("instruction", ""),
                    episode_id=task.get("episode_id", f"ep_{time.time_ns()}"),
                    benchmark="webshop",
                    y_w_tokens=low_k["tokens"],
                    y_w_k_used=low_k["k_used"],
                    y_w_reward=low_k["reward"],
                    y_w_success=True,
                    y_l_tokens=high_k["tokens"],
                    y_l_k_used=high_k["k_used"],
                    y_l_reward=high_k["reward"],
                    y_l_success=True,
                    context_tokens=ctx_tokens,
                ))

        return pairs


# ── DPO Rollout Trainer ───────────────────────────────────────────────────────

@dataclass
class DPOConfig:
    # Data
    preference_files: List[str] = field(default_factory=list)
    # also support live rollout collection
    use_live_rollouts: bool = False
    rollout_envs: List[str] = field(default_factory=lambda: ["webshop"])

    # Training
    num_epochs: int = 1
    batch_size: int = 2
    gradient_accumulation_steps: int = 16
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 50
    max_grad_norm: float = 1.0
    max_seq_len: int = 1024

    # DPO
    beta: float = 0.1
    bandwidth_weight: float = 0.01

    # Checkpoint to start from
    sft_checkpoint: Optional[str] = None

    # Logging
    output_dir: str = "outputs/dpo"
    log_every: int = 10
    save_every: int = 200
    use_wandb: bool = False
    project_name: str = "latent_agent_team"
    run_name: str = "dpo_rollout"


class DPORollout:
    """
    Stage 2: DPO preference optimization for success-minus-bandwidth objective.
    """

    def __init__(
        self,
        cfg: DPOConfig,
        policy_backbone: BackboneManager,
        ref_backbone: BackboneManager,   # frozen reference policy
        role_manager: RoleAdapterManager,
        comm_module: LatentCommunicationModule,
        bitrate_scheduler: AdaptiveBitrateScheduler,
    ):
        self.cfg = cfg
        self.policy = policy_backbone
        self.ref = ref_backbone
        self.role_manager = role_manager
        self.comm_module = comm_module
        self.bitrate_scheduler = bitrate_scheduler

        os.makedirs(cfg.output_dir, exist_ok=True)

    def _compute_log_probs(
        self,
        backbone: BackboneManager,
        context_ids: torch.Tensor,      # [B, T_ctx]
        response_ids: torch.Tensor,     # [B, T_resp]
        response_mask: torch.Tensor,    # [B, T_resp]
    ) -> torch.Tensor:
        """
        Compute per-sequence log probabilities for DPO.
        Returns [B] — sum of token log probs for each sequence.
        """
        # Concatenate context + response
        full_ids = torch.cat([context_ids, response_ids], dim=1)
        full_mask = torch.cat([
            torch.ones_like(context_ids),
            response_mask,
        ], dim=1)

        with torch.no_grad() if backbone is self.ref else torch.enable_grad():
            outputs = backbone.model(
                input_ids=full_ids,
                attention_mask=full_mask,
            )

        logits = outputs.logits  # [B, T, vocab]
        ctx_len = context_ids.size(1)
        resp_len = response_ids.size(1)

        # Only compute log probs over the response tokens
        resp_logits = logits[:, ctx_len - 1 : ctx_len + resp_len - 1, :]  # [B, T_resp, V]
        log_probs = F.log_softmax(resp_logits, dim=-1)  # [B, T_resp, V]

        # Gather log probs of actual tokens
        token_logps = log_probs.gather(
            -1, response_ids.unsqueeze(-1)
        ).squeeze(-1)  # [B, T_resp]

        # Mask padding
        token_logps = token_logps * response_mask.float()

        # Sum over sequence length
        return token_logps.sum(dim=-1)  # [B]

    def train_step(
        self,
        batch: Dict[str, Any],
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> Dict[str, float]:
        context_ids = batch["context_ids"].to(device)
        yw_ids = batch["yw_ids"].to(device)
        yw_mask = batch["yw_mask"].to(device)
        yl_ids = batch["yl_ids"].to(device)
        yl_mask = batch["yl_mask"].to(device)
        yw_k = batch["yw_k"].to(device)

        # ── Policy log probs ─────────────────────────────────────────────
        policy_yw_logps = self._compute_log_probs(self.policy, context_ids, yw_ids, yw_mask)
        policy_yl_logps = self._compute_log_probs(self.policy, context_ids, yl_ids, yl_mask)

        # ── Reference log probs (no grad) ────────────────────────────────
        ref_yw_logps = self._compute_log_probs(self.ref, context_ids, yw_ids, yw_mask)
        ref_yl_logps = self._compute_log_probs(self.ref, context_ids, yl_ids, yl_mask)

        # ── Bandwidth penalty ────────────────────────────────────────────
        bandwidth_penalty = yw_k / 64.0  # normalize

        # ── DPO loss ─────────────────────────────────────────────────────
        loss, info = dpo_loss(
            policy_yw_logps=policy_yw_logps,
            policy_yl_logps=policy_yl_logps,
            ref_yw_logps=ref_yw_logps,
            ref_yl_logps=ref_yl_logps,
            beta=self.cfg.beta,
            bandwidth_penalty=bandwidth_penalty,
            bandwidth_weight=self.cfg.bandwidth_weight,
        )

        loss.backward()
        return info

    def train(self, device: Optional[torch.device] = None) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Starting DPO rollout training on {device}")

        # Collect trainable params (policy only, not reference)
        trainable = [p for p in self.policy.model.parameters() if p.requires_grad]
        if self.comm_module.channel is not None:
            trainable.extend(self.comm_module.channel.parameters())
        trainable.extend(self.bitrate_scheduler.parameters())

        optimizer = torch.optim.AdamW(
            trainable,
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )

        dataset = PreferenceDataset(
            preference_files=self.cfg.preference_files,
            tokenizer=self.policy.tokenizer,
            max_seq_len=self.cfg.max_seq_len,
        )

        if len(dataset) == 0:
            logger.warning("No preference data loaded. Skipping DPO training.")
            return

        dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=2,
            collate_fn=PreferenceDataset.collate_fn,
        )

        global_step = 0
        for epoch in range(self.cfg.num_epochs):
            for step, batch in enumerate(dataloader):
                info = self.train_step(batch, optimizer, device)

                if (step + 1) % self.cfg.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(trainable, self.cfg.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if global_step % self.cfg.log_every == 0:
                        logger.info(
                            f"DPO Step {global_step} | "
                            f"Loss: {info.get('total', 0):.4f} | "
                            f"Margin: {info.get('reward_margin', 0):.4f}"
                        )

                    if global_step % self.cfg.save_every == 0:
                        ckpt_dir = os.path.join(self.cfg.output_dir, f"checkpoint_{global_step}")
                        os.makedirs(ckpt_dir, exist_ok=True)
                        torch.save(
                            self.comm_module.state_dict(),
                            os.path.join(ckpt_dir, "comm_module.pt"),
                        )
                        logger.info(f"DPO checkpoint saved: {ckpt_dir}")

        final_dir = os.path.join(self.cfg.output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        if self.comm_module.channel is not None:
            torch.save(
                self.comm_module.state_dict(),
                os.path.join(final_dir, "comm_module.pt"),
            )
        logger.info(f"DPO training complete.")


def _generate_synthetic_preferences(
    team,
    tokenizer,
    test_data: list,
    k_choices: list = None,
    max_tasks: int = 50,
    output_file: str = "preferences.jsonl",
) -> str:
    """
    Generate preference pairs by rolling out the team at different K budgets.
    For each task, run at K_low and K_high, prefer the better outcome.
    """
    if k_choices is None:
        k_choices = [4, 8, 32, 64]
    pairs = []
    device = team.device

    for ti, task in enumerate(test_data[:max_tasks]):
        instruction = task.get("instruction", task.get("confirmed_task", ""))
        if not instruction:
            continue
        ctx_tokens = tokenizer.encode(instruction[:200], add_special_tokens=False)[:128]

        trajectories = []
        for k_val in k_choices:
            team.reset(task_instruction=instruction)
            traj_tokens = []
            total_reward = 0.0
            success = False

            # Run a few steps
            actions_list = task.get("actions", [])
            for ai, action_gt in enumerate(actions_list[:5]):
                html = str(action_gt.get("cleaned_html", ""))[:800]
                obs = f"Task: {instruction}\nHTML:\n{html}"
                try:
                    pred_action, info = team.step(obs, instruction, k_budget=k_val)
                    action_str = json.dumps(pred_action) if isinstance(pred_action, dict) else str(pred_action)
                    step_tokens = tokenizer.encode(action_str, add_special_tokens=False)[:32]
                    traj_tokens.extend(step_tokens)

                    # Check if prediction matches ground truth
                    gt_op = action_gt.get("operation", {}).get("op", "CLICK").upper()
                    pred_op = str(pred_action.get("action", "")).upper()
                    if gt_op in pred_op or pred_op in gt_op:
                        total_reward += 0.5
                        success = True
                except Exception:
                    traj_tokens.extend(tokenizer.encode("error", add_special_tokens=False))

            trajectories.append({
                "tokens": traj_tokens[:256],
                "k_used": k_val,
                "reward": total_reward,
                "success": success,
            })

        # Build preference pairs
        success_trajs = [t for t in trajectories if t["success"]]
        fail_trajs = [t for t in trajectories if not t["success"]]

        if success_trajs and fail_trajs:
            best = min(success_trajs, key=lambda t: t["k_used"])
            worst = max(fail_trajs, key=lambda t: t["k_used"])
            pairs.append({
                "task_instruction": instruction[:200],
                "episode_id": f"ep_{ti}",
                "benchmark": "mind2web",
                "y_w_tokens": best["tokens"],
                "y_w_k_used": best["k_used"],
                "y_w_reward": best["reward"],
                "y_w_success": True,
                "y_l_tokens": worst["tokens"],
                "y_l_k_used": worst["k_used"],
                "y_l_reward": worst["reward"],
                "y_l_success": False,
                "context_tokens": ctx_tokens,
            })
        elif len(success_trajs) >= 2:
            sorted_trajs = sorted(success_trajs, key=lambda t: t["k_used"])
            lo, hi = sorted_trajs[0], sorted_trajs[-1]
            if lo["k_used"] < hi["k_used"]:
                pairs.append({
                    "task_instruction": instruction[:200],
                    "episode_id": f"ep_{ti}",
                    "benchmark": "mind2web",
                    "y_w_tokens": lo["tokens"],
                    "y_w_k_used": lo["k_used"],
                    "y_w_reward": lo["reward"],
                    "y_w_success": True,
                    "y_l_tokens": hi["tokens"],
                    "y_l_k_used": hi["k_used"],
                    "y_l_reward": hi["reward"],
                    "y_l_success": True,
                    "context_tokens": ctx_tokens,
                })

    with open(output_file, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")
    logger.info(f"Generated {len(pairs)} preference pairs → {output_file}")
    return output_file


def main():
    """CLI entry point for DPO training."""
    import argparse
    import copy
    parser = argparse.ArgumentParser(description="Stage 2: DPO Rollout")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--sft_checkpoint", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/dpo")
    parser.add_argument("--backbone", type=str, default="llama32_3b")
    parser.add_argument("--comm_mode", type=str, default="text")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--generate_prefs", action="store_true",
                        help="Generate preference pairs from rollouts before training")
    parser.add_argument("--max_pref_tasks", type=int, default=50)
    args = parser.parse_args()

    from omegaconf import OmegaConf

    cfg_dict = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)

    # Extract DPO-specific fields
    dpo_fields = {k: v for k, v in cfg_dict.get("dpo", {}).items()
                  if k in DPOConfig.__dataclass_fields__}
    dpo_cfg = DPOConfig(
        output_dir=args.output_dir,
        sft_checkpoint=args.sft_checkpoint,
        **dpo_fields,
    )

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info(f"DPO config: {dpo_cfg}")
    os.makedirs(dpo_cfg.output_dir, exist_ok=True)

    # ── Build policy team ────────────────────────────────────────────────
    from ..team import AgentTeam

    cfg_dict["backbone"]["device_map"] = str(device)
    cfg_dict["communication"]["mode"] = args.comm_mode
    cfg_dict["backbone"]["name"] = args.backbone
    cfg_dict["backbone"]["gradient_checkpointing"] = True

    logger.info(f"Building policy team ({args.backbone}/{args.comm_mode})...")
    policy_team = AgentTeam.from_config(cfg_dict, device=device)

    # Load SFT checkpoint if provided
    if args.sft_checkpoint and os.path.exists(args.sft_checkpoint):
        comm_path = os.path.join(args.sft_checkpoint, "comm_module.pt")
        if os.path.exists(comm_path):
            policy_team.comm.load_state_dict(
                torch.load(comm_path, map_location=device), strict=False)
            logger.info(f"Loaded SFT checkpoint from {args.sft_checkpoint}")

    # ── Build reference model (frozen copy of policy) ────────────────────
    # For memory efficiency, we use the same backbone with frozen weights
    # by computing reference log probs with torch.no_grad()
    from ..models.backbone import BackboneManager, BackboneConfig
    logger.info("Building reference model (frozen)...")
    ref_backbone = copy.deepcopy(policy_team.backbone)
    ref_backbone.model.eval()
    for p in ref_backbone.model.parameters():
        p.requires_grad = False

    # ── Generate preference data if needed ───────────────────────────────
    pref_file = os.path.join(dpo_cfg.output_dir, "preferences.jsonl")
    if args.generate_prefs or not dpo_cfg.preference_files:
        logger.info("Generating preference pairs from rollouts...")
        try:
            from datasets import load_dataset
            ds = load_dataset("osunlp/Mind2Web", split="train", streaming=True)
            tasks = []
            for sample in ds:
                task = {
                    "instruction": sample["confirmed_task"],
                    "actions": [],
                }
                for act in sample.get("actions", []):
                    task["actions"].append({
                        "operation": act.get("operation", {}),
                        "cleaned_html": str(act.get("cleaned_html", ""))[:1500],
                        "pos_candidates": act.get("pos_candidates", [])[:3],
                    })
                tasks.append(task)
                if len(tasks) >= args.max_pref_tasks:
                    break

            pref_file = _generate_synthetic_preferences(
                team=policy_team,
                tokenizer=policy_team.backbone.tokenizer,
                test_data=tasks,
                max_tasks=args.max_pref_tasks,
                output_file=pref_file,
            )
        except Exception as e:
            logger.error(f"Failed to generate preferences: {e}")
            return

        dpo_cfg.preference_files = [pref_file]

    if not dpo_cfg.preference_files:
        dpo_cfg.preference_files = [pref_file]

    # ── Run DPO training ─────────────────────────────────────────────────
    dpo_trainer = DPORollout(
        cfg=dpo_cfg,
        policy_backbone=policy_team.backbone,
        ref_backbone=ref_backbone,
        role_manager=policy_team.role_manager,
        comm_module=policy_team.comm,
        bitrate_scheduler=policy_team.bitrate,
    )

    logger.info("Starting DPO training...")
    dpo_trainer.train(device=device)

    # Save final checkpoint
    final_dir = os.path.join(dpo_cfg.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    policy_team.save_checkpoint(final_dir)
    logger.info(f"DPO training complete. Checkpoint saved to {final_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
