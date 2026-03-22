#!/usr/bin/env python3
"""
run_sft_training.py — End-to-end SFT training launcher.

Builds the full AgentTeam components from a config YAML, loads teacher
trace files from data/traces/, and runs Stage 1 supervised bootstrapping.

Usage:
    # Single backbone + comm mode
    python scripts/run_sft_training.py --backbone phi3_mini --comm_mode continuous --gpu 0

    # With specific trace files
    python scripts/run_sft_training.py --backbone phi3_mini --traces data/traces/traces_phi3_mini.jsonl

    # Quick sanity run
    python scripts/run_sft_training.py --backbone phi3_mini --quick
"""

from __future__ import annotations

import gc
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from latent_agent_team.models.backbone import BackboneManager, BackboneConfig, BACKBONE_REGISTRY
from latent_agent_team.models.role_adapter import RoleAdapterManager
from latent_agent_team.models.latent_comm import LatentCommunicationModule, CommModuleConfig
from latent_agent_team.models.router import AdaptiveBitrateScheduler, SparseRouter
from latent_agent_team.models.audit_decoder import AuditDecoder
from latent_agent_team.train.sft_bootstrap import (
    SFTBootstrapper, SFTConfig, LossWeights, TeacherTraceDataset,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("sft_training")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"
TRACE_DIR = PROJECT_ROOT / "data" / "traces"

# Backbone name → config file
CONFIG_MAP = {
    "phi3_mini": "phi3.yaml",
    "llama32_3b": "llama32_3b.yaml",
    "gemma2_9b": "gemma2_9b.yaml",
    "ministral_8b": "ministral3_8b.yaml",
}


def build_components_from_config(
    cfg: Dict[str, Any],
    comm_mode: str,
    device: torch.device,
):
    """Build all trainable components from a config dict."""
    bb_cfg = cfg["backbone"]
    backbone_name = bb_cfg.get("name", "phi3_mini")

    # ── Backbone ────────────────────────────────────────────────────────
    lora_cfg = cfg.get("lora", {})
    backbone_cfg = BackboneConfig(
        backbone_name=backbone_name,
        quantization=bb_cfg.get("quantization", "4bit"),
        max_seq_len=bb_cfg.get("max_seq_len", 4096),
        dtype=bb_cfg.get("dtype", "bfloat16"),
        use_flash_attention=bb_cfg.get("use_flash_attention", True),
        gradient_checkpointing=bb_cfg.get("gradient_checkpointing", True),
        device_map=str(device),
    )
    backbone_cfg.lora_r = lora_cfg.get("r", 16)
    backbone_cfg.lora_alpha = lora_cfg.get("alpha", 32)
    backbone_cfg.lora_dropout = lora_cfg.get("dropout", 0.05)

    backbone = BackboneManager(backbone_cfg)

    roles = ["planner", "retriever", "browser", "verifier", "memory"]
    for role in roles:
        backbone.add_adapter(role)

    hidden_size = backbone.hidden_size

    # ── Role Adapters ──────────────────────────────────────────────────
    role_manager = RoleAdapterManager(hidden_size=hidden_size, roles=roles)
    role_manager.to(device)

    # ── Communication Module ───────────────────────────────────────────
    comm_cfg_dict = cfg.get("communication", {})
    comm_config = CommModuleConfig(
        mode=comm_mode,
        hidden_size=hidden_size,
        latent_dim=comm_cfg_dict.get("latent_dim", 256),
        codebook_size=comm_cfg_dict.get("codebook_size", 512),
        commitment_cost=comm_cfg_dict.get("commitment_cost", 0.25),
        ema_decay=comm_cfg_dict.get("ema_decay", 0.99),
        use_gumbel_warmup=comm_cfg_dict.get("use_gumbel_warmup", True),
        injection_mode=comm_cfg_dict.get("injection_mode", "prefix"),
    )
    comm_module = LatentCommunicationModule(comm_config).to(device)

    # ── Bitrate Scheduler ──────────────────────────────────────────────
    bs_cfg = cfg.get("bitrate_scheduler", {})
    bitrate_scheduler = AdaptiveBitrateScheduler(
        hidden_dim=bs_cfg.get("hidden_dim", 64),
    ).to(device)

    # ── Sparse Router ──────────────────────────────────────────────────
    sr_cfg = cfg.get("sparse_router", {})
    router = SparseRouter(
        hidden_size=hidden_size,
        router_hidden_dim=sr_cfg.get("router_hidden_dim", 128),
        routing_threshold=sr_cfg.get("routing_threshold", 0.5),
        min_recipients=sr_cfg.get("min_recipients", 1),
    ).to(device)

    # ── Audit Decoder ──────────────────────────────────────────
    ad_cfg = cfg.get("audit_decoder", {})
    # Use model config vocab_size (includes added/special tokens + padding)
    effective_vocab_size = backbone.model.config.vocab_size
    audit_decoder = AuditDecoder(
        hidden_size=hidden_size,
        vocab_size=effective_vocab_size,
        decoder_dim=ad_cfg.get("decoder_dim", 256),
        max_decode_len=ad_cfg.get("max_decode_len", 32),
    ).to(device)

    return backbone, role_manager, comm_module, bitrate_scheduler, router, audit_decoder


def build_sft_config(
    cfg: Dict[str, Any],
    comm_mode: str,
    backbone_name: str,
    trace_files: List[str],
    output_dir: str,
    quick: bool = False,
) -> SFTConfig:
    """Build SFTConfig from YAML config + CLI args."""
    sft_yaml = cfg.get("sft", {})
    loss_w = sft_yaml.get("loss_weights", {})

    sft_cfg = SFTConfig(
        trace_files=trace_files,
        benchmarks=["mind2web", "webshop", "agentbench"],
        num_epochs=1 if quick else sft_yaml.get("num_epochs", 3),
        batch_size=2,  # Reduced to fit in GPU memory with training activations
        gradient_accumulation_steps=sft_yaml.get("gradient_accumulation_steps", 8),
        learning_rate=sft_yaml.get("learning_rate", 2e-4),
        weight_decay=sft_yaml.get("weight_decay", 0.01),
        warmup_steps=10 if quick else sft_yaml.get("warmup_steps", 100),
        max_grad_norm=sft_yaml.get("max_grad_norm", 1.0),
        max_seq_len=512,  # Reduced from 2048 to fit training in ~40GB VRAM
        comm_mode=comm_mode,
        k_initial=cfg.get("communication", {}).get("k_initial", 8),
        output_dir=output_dir,
        log_every=5 if quick else 10,
        save_every=50 if quick else 500,
        use_wandb=False,
        project_name="latent_agent_team",
        run_name=f"sft_{backbone_name}_{comm_mode}",
        loss_weights=LossWeights(
            next_action_ce=loss_w.get("next_action_ce", 1.0),
            constraint_recon=loss_w.get("constraint_recon", 0.3),
            audit_decode=loss_w.get("audit_decode", 0.1),
            routing_supervision=loss_w.get("routing_supervision", 0.2),
            bitrate_reg=loss_w.get("bitrate_reg", 0.01),
            vq_commitment=loss_w.get("vq_commitment", 1.0),
        ),
    )
    return sft_cfg


def find_trace_files(backbone_name: str) -> List[str]:
    """Find all trace JSONL files for a given backbone."""
    trace_dir = TRACE_DIR
    candidates = [
        trace_dir / f"traces_{backbone_name}.jsonl",
        trace_dir / f"traces_all.jsonl",
    ]
    # Also check any file that matches
    if trace_dir.exists():
        for f in trace_dir.glob("traces_*.jsonl"):
            if f not in candidates:
                candidates.append(f)

    found = [str(f) for f in candidates if f.exists()]
    if not found:
        logger.warning(f"No trace files found for {backbone_name} in {trace_dir}")
    return found


def run_single_training(
    backbone_name: str,
    comm_mode: str,
    gpu_id: int,
    trace_files: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    quick: bool = False,
) -> Dict[str, Any]:
    """Run SFT training for a single backbone + comm_mode combination."""
    t0 = time.time()
    device = torch.device(f"cuda:{gpu_id}")

    logger.info(f"\n{'='*70}")
    logger.info(f"SFT Training: {backbone_name} / {comm_mode} on GPU {gpu_id}")
    logger.info(f"{'='*70}")

    # Load config
    config_file = CONFIGS_DIR / CONFIG_MAP[backbone_name]
    cfg = OmegaConf.to_container(OmegaConf.load(config_file), resolve=True)

    # Find trace files
    if trace_files is None:
        trace_files = find_trace_files(backbone_name)
    if not trace_files:
        return {"backbone": backbone_name, "comm_mode": comm_mode, "status": "no_traces"}

    logger.info(f"Using trace files: {trace_files}")

    # Build output dir
    if output_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = str(PROJECT_ROOT / "outputs" / f"sft_{backbone_name}_{comm_mode}_{timestamp}")

    # Build components
    logger.info("Building model components...")
    backbone, role_manager, comm_module, bitrate_scheduler, router, audit_decoder = \
        build_components_from_config(cfg, comm_mode, device)

    # Build SFT config
    sft_cfg = build_sft_config(cfg, comm_mode, backbone_name, trace_files, output_dir, quick)

    logger.info(f"SFT config: epochs={sft_cfg.num_epochs}, batch={sft_cfg.batch_size}, "
                f"lr={sft_cfg.learning_rate}, grad_accum={sft_cfg.gradient_accumulation_steps}")

    # Verify trace data loads
    test_ds = TeacherTraceDataset(
        trace_files=trace_files,
        tokenizer=backbone.tokenizer,
        max_seq_len=sft_cfg.max_seq_len,
    )
    logger.info(f"Loaded {len(test_ds)} trace steps")
    if len(test_ds) == 0:
        return {"backbone": backbone_name, "comm_mode": comm_mode, "status": "empty_traces"}
    del test_ds

    # Build trainer
    trainer = SFTBootstrapper(
        cfg=sft_cfg,
        backbone=backbone,
        role_manager=role_manager,
        comm_module=comm_module,
        bitrate_scheduler=bitrate_scheduler,
        sparse_router=router,
        audit_decoder=audit_decoder,
    )

    # Train
    logger.info("Starting training...")
    trainer.train(device=device)

    elapsed = time.time() - t0
    result = {
        "backbone": backbone_name,
        "comm_mode": comm_mode,
        "status": "complete",
        "output_dir": output_dir,
        "elapsed_sec": elapsed,
        "elapsed_human": f"{elapsed/60:.1f} min",
    }

    # Cleanup GPU memory
    del trainer, backbone, role_manager, comm_module, bitrate_scheduler, router, audit_decoder
    gc.collect()
    torch.cuda.empty_cache()

    logger.info(f"Training complete in {elapsed/60:.1f} min. Checkpoint: {output_dir}")
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run SFT Training")
    parser.add_argument("--backbone", type=str, default="phi3_mini",
                        choices=list(CONFIG_MAP.keys()))
    parser.add_argument("--comm_mode", type=str, default="continuous",
                        choices=["continuous", "vq", "text"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--traces", type=str, nargs="+", default=None,
                        help="Explicit trace file paths")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--quick", action="store_true",
                        help="Quick run: 1 epoch, fewer warmup steps")
    args = parser.parse_args()

    result = run_single_training(
        backbone_name=args.backbone,
        comm_mode=args.comm_mode,
        gpu_id=args.gpu,
        trace_files=args.traces,
        output_dir=args.output_dir,
        quick=args.quick,
    )

    logger.info(f"\nResult: {json.dumps(result, indent=2)}")


if __name__ == "__main__":
    main()
