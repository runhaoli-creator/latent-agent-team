#!/usr/bin/env python3
"""
Quick evaluation of fine-tuned checkpoints on Mind2Web/WebLINX/AgentInstruct.
Loads the benchmark-fine-tuned checkpoint instead of the base SFT checkpoint.
"""
import argparse
import gc
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import torch
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("eval_finetuned")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_real_experiments import (
    build_team, load_checkpoint, 
    load_mind2web_data, load_weblinx_data, load_agentinstruct_data,
    evaluate_mind2web, evaluate_weblinx, evaluate_agentinstruct,
)


def load_finetuned_checkpoint(team, ft_dir: str, backbone: str, comm: str, benchmark: str = "mind2web"):
    """Load a fine-tuned checkpoint from benchmark FT output.
    
    FT checkpoints are at: ft_dir/ft_{backbone}_{comm}/checkpoint_{benchmark}/
    We reuse load_checkpoint by creating a temp symlink structure.
    """
    ft_ckpt = Path(ft_dir) / f"ft_{backbone}_{comm}" / f"checkpoint_{benchmark}"
    if not ft_ckpt.exists():
        # Try final checkpoint
        ft_ckpt = Path(ft_dir) / f"ft_{backbone}_{comm}" / "final"
    if not ft_ckpt.exists():
        logger.warning(f"No fine-tuned checkpoint at {ft_ckpt}")
        return False
    
    # Create temporary sft-compatible directory structure
    tmp_sft_dir = Path(ft_dir) / "_tmp_eval"
    tmp_sft_dir.mkdir(exist_ok=True)
    sft_link = tmp_sft_dir / f"sft_{backbone}_{comm}" / "final"
    sft_link.parent.mkdir(parents=True, exist_ok=True)
    
    # Remove old symlink if exists
    if sft_link.exists() or sft_link.is_symlink():
        sft_link.unlink()
    
    sft_link.symlink_to(ft_ckpt.resolve())
    
    result = load_checkpoint(team, str(tmp_sft_dir), backbone, comm)
    
    # Cleanup
    sft_link.unlink()
    
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="llama32_3b")
    parser.add_argument("--comm_mode", default="text")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--ft_dir", required=True, help="Fine-tuning output directory")
    parser.add_argument("--benchmark", default="mind2web", help="Which benchmark checkpoint to load")
    parser.add_argument("--eval_benchmarks", nargs="+", default=["mind2web"], 
                       help="Which benchmarks to evaluate on")
    parser.add_argument("--max_tasks", type=int, default=200, help="Max tasks to evaluate")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or str(PROJECT_ROOT / f"outputs/ft_eval_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Building team: {args.backbone}/{args.comm_mode}")
    team = build_team(args.backbone, args.comm_mode, device)
    
    # First load base SFT checkpoint
    base_sft_dir = str(PROJECT_ROOT / "outputs/sft_all_20260309_110924")
    if os.path.exists(base_sft_dir):
        logger.info(f"Loading base SFT checkpoint...")
        load_checkpoint(team, base_sft_dir, args.backbone, args.comm_mode)
    
    # Then load fine-tuned checkpoint on top
    logger.info(f"Loading fine-tuned checkpoint from {args.ft_dir} (benchmark={args.benchmark})")
    load_finetuned_checkpoint(team, args.ft_dir, args.backbone, args.comm_mode, args.benchmark)
    
    team.backbone.model.eval()
    
    if args.fast and hasattr(team, 'enable_fast_eval'):
        team.enable_fast_eval()
    
    all_results = {}
    
    for bench in args.eval_benchmarks:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating {args.backbone}/{args.comm_mode} on {bench}")
        logger.info(f"{'='*60}")
        
        if bench == "mind2web":
            m2w_test = load_mind2web_data(max_tasks=args.max_tasks)
            with torch.no_grad():
                results = evaluate_mind2web(team, m2w_test, max_steps=20)
            logger.info(f"Mind2Web: elem_acc={results['elem_acc']:.4f} op_f1={results['op_f1']:.4f} step_sr={results['step_sr']:.4f}")
            all_results["mind2web"] = results
            
        elif bench == "weblinx":
            wl_test = load_weblinx_data(max_samples=args.max_tasks)
            with torch.no_grad():
                results = evaluate_weblinx(team, wl_test)
            logger.info(f"WebLINX: type_acc={results['type_acc']:.4f} exact_match={results['exact_match']:.4f}")
            all_results["weblinx"] = results
            
        elif bench.startswith("agentinstruct"):
            ai_test = load_agentinstruct_data(max_tasks=args.max_tasks)
            with torch.no_grad():
                results = evaluate_agentinstruct(team, ai_test)
            logger.info(f"AgentInstruct: bleu={results.get('bleu',0):.4f}")
            all_results["agentinstruct"] = results
    
    # Save results
    result_file = os.path.join(output_dir, f"ft_eval_{args.backbone}_{args.comm_mode}.json")
    with open(result_file, "w") as f:
        json.dump({
            "backbone": args.backbone,
            "comm_mode": args.comm_mode,
            "ft_dir": args.ft_dir,
            "benchmark_ckpt": args.benchmark,
            "eval_benchmarks": args.eval_benchmarks,
            "results": {k: {kk: vv for kk, vv in v.items() if kk != "per_task" and kk != "per_sample"} 
                       for k, v in all_results.items()},
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)
    
    logger.info(f"\nResults saved to {result_file}")
    
    # Print summary
    logger.info(f"\n{'='*70}")
    logger.info(f"FINE-TUNED EVALUATION SUMMARY: {args.backbone}/{args.comm_mode}")
    logger.info(f"{'='*70}")
    for bench, res in all_results.items():
        if bench == "mind2web":
            logger.info(f"  Mind2Web: elem_acc={res['elem_acc']*100:.1f}% op_f1={res['op_f1']*100:.1f}% step_sr={res['step_sr']*100:.1f}%")
        elif bench == "weblinx":
            logger.info(f"  WebLINX: type_acc={res['type_acc']*100:.1f}% exact_match={res['exact_match']*100:.1f}% intent={res['intent_match']*100:.1f}%")
        elif bench == "agentinstruct":
            for env_key, env_res in res.items():
                if isinstance(env_res, dict) and "bleu" in env_res:
                    logger.info(f"  AgentInstruct/{env_key}: bleu={env_res['bleu']:.4f} rouge_l={env_res.get('rouge_l',0):.4f}")
    
    del team
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
