#!/usr/bin/env python3
"""
run_full_sft_pipeline.py — Full SFT training pipeline orchestrator.

Phase A: Generate teacher traces for all 4 backbones (parallel on 4 GPUs)
Phase B: Train all 12 configs (4 backbones × 3 comm modes) across 8 GPUs
Phase C: Re-run main experiments with trained checkpoints
Phase D: Generate results report

Usage:
    nohup python scripts/run_full_sft_pipeline.py > outputs/sft_pipeline.log 2>&1 &

    # Quick smoke test
    python scripts/run_full_sft_pipeline.py --quick --backbones phi3_mini --comm_modes continuous
"""

from __future__ import annotations

import gc
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("sft_pipeline")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRACE_DIR = PROJECT_ROOT / "data" / "traces"
CONFIGS_DIR = PROJECT_ROOT / "configs"

ALL_BACKBONES = ["phi3_mini", "llama32_3b", "gemma2_9b", "ministral_8b"]
ALL_COMM_MODES = ["continuous", "vq", "text"]
CONFIG_MAP = {
    "phi3_mini": "phi3.yaml",
    "llama32_3b": "llama32_3b.yaml",
    "gemma2_9b": "gemma2_9b.yaml",
    "ministral_8b": "ministral3_8b.yaml",
}

# GPU assignment: 8 GPUs, assign 2 training jobs per GPU if needed
# Small models (phi3, llama32) → GPUs 0-3
# Large models (gemma2, ministral) → GPUs 4-7
GPU_ASSIGNMENTS = {
    "phi3_mini":    [0, 1],   # 2 GPUs for phi3 (3 comm modes → 2 parallel + 1 serial)
    "llama32_3b":   [2, 3],
    "gemma2_9b":    [4, 5],
    "ministral_8b": [6, 7],
}


def run_trace_generation(backbone: str, gpu: int, n_tasks: int = 200, no_model: bool = False) -> Dict:
    """Run trace generation for one backbone (subprocess)."""
    cmd = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "generate_teacher_traces.py"),
        "--backbone", backbone,
        "--gpu", str(gpu),
        "--n_webshop", str(n_tasks),
        "--n_mind2web", str(n_tasks),
        "--n_agentbench", str(n_tasks // 2),
    ]
    if no_model:
        cmd.append("--no_model")

    logger.info(f"[TRACE] Starting trace gen: {backbone} on GPU {gpu}")
    t0 = time.time()

    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=7200,
        cwd=str(PROJECT_ROOT),
    )

    elapsed = time.time() - t0
    trace_file = str(TRACE_DIR / f"traces_{backbone}.jsonl")
    success = result.returncode == 0 and os.path.exists(trace_file)

    if not success:
        logger.error(f"[TRACE] FAILED {backbone}: {result.stderr[-500:]}")
    else:
        # Count trace lines
        with open(trace_file) as f:
            n_lines = sum(1 for _ in f)
        logger.info(f"[TRACE] Done {backbone}: {n_lines} traces in {elapsed/60:.1f} min")

    return {
        "backbone": backbone,
        "status": "ok" if success else "failed",
        "trace_file": trace_file if success else None,
        "elapsed_sec": elapsed,
        "stderr": result.stderr[-500:] if not success else "",
    }


def run_sft_training(
    backbone: str, comm_mode: str, gpu: int,
    trace_files: List[str], output_dir: str, quick: bool = False,
) -> Dict:
    """Run SFT training for one backbone+comm_mode (subprocess)."""
    cmd = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "run_sft_training.py"),
        "--backbone", backbone,
        "--comm_mode", comm_mode,
        "--gpu", str(gpu),
        "--traces", *trace_files,
        "--output_dir", output_dir,
    ]
    if quick:
        cmd.append("--quick")

    logger.info(f"[SFT] Starting: {backbone}/{comm_mode} on GPU {gpu}")
    t0 = time.time()

    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=14400,  # 4 hour timeout
        cwd=str(PROJECT_ROOT),
    )

    elapsed = time.time() - t0
    success = result.returncode == 0

    # Check if checkpoint exists
    ckpt_dir = os.path.join(output_dir, "final")
    has_checkpoint = os.path.exists(ckpt_dir)

    if not success:
        logger.error(f"[SFT] FAILED {backbone}/{comm_mode}: {result.stderr[-500:]}")
    else:
        logger.info(f"[SFT] Done {backbone}/{comm_mode} in {elapsed/60:.1f} min")

    return {
        "backbone": backbone,
        "comm_mode": comm_mode,
        "status": "ok" if success and has_checkpoint else "failed",
        "output_dir": output_dir,
        "has_checkpoint": has_checkpoint,
        "elapsed_sec": elapsed,
        "stderr": result.stderr[-500:] if not success else "",
    }


def phase_a_traces(
    backbones: List[str],
    n_tasks: int = 200,
    no_model: bool = False,
) -> Dict[str, str]:
    """Phase A: Generate teacher traces for all backbones in parallel."""
    logger.info("\n" + "=" * 70)
    logger.info("PHASE A: Generating Teacher Traces")
    logger.info("=" * 70)

    TRACE_DIR.mkdir(parents=True, exist_ok=True)

    # Check for existing traces
    trace_files = {}
    to_generate = []
    for bb in backbones:
        trace_file = str(TRACE_DIR / f"traces_{bb}.jsonl")
        if os.path.exists(trace_file):
            with open(trace_file) as f:
                n_lines = sum(1 for _ in f)
            if n_lines > 100:
                logger.info(f"  [SKIP] {bb}: using existing {trace_file} ({n_lines} traces)")
                trace_files[bb] = trace_file
                continue
        to_generate.append(bb)

    if not to_generate:
        logger.info("All trace files already exist. Skipping Phase A.")
        return trace_files

    # Generate traces in parallel (one per GPU)
    results = []
    with ProcessPoolExecutor(max_workers=min(len(to_generate), 4)) as executor:
        futures = {}
        for i, bb in enumerate(to_generate):
            gpu = GPU_ASSIGNMENTS[bb][0]
            fut = executor.submit(run_trace_generation, bb, gpu, n_tasks, no_model)
            futures[fut] = bb

        for fut in as_completed(futures):
            bb = futures[fut]
            try:
                res = fut.result()
                results.append(res)
                if res["status"] == "ok":
                    trace_files[bb] = res["trace_file"]
            except Exception as e:
                logger.error(f"[TRACE] Exception for {bb}: {e}")

    logger.info(f"\nPhase A complete: {len(trace_files)}/{len(backbones)} trace files ready")
    return trace_files


def phase_b_training(
    backbones: List[str],
    comm_modes: List[str],
    trace_files: Dict[str, str],
    output_base: str,
    quick: bool = False,
) -> List[Dict]:
    """
    Phase B: Run SFT training for all backbone × comm_mode combos.
    
    Runs up to 2 parallel trainings per backbone (on its assigned GPUs).
    """
    logger.info("\n" + "=" * 70)
    logger.info("PHASE B: SFT Training (all configurations)")
    logger.info("=" * 70)

    # Build job list
    jobs = []
    for bb in backbones:
        if bb not in trace_files:
            logger.warning(f"  Skipping {bb}: no trace file")
            continue
        for cm in comm_modes:
            output_dir = os.path.join(output_base, f"sft_{bb}_{cm}")
            # Check if already trained
            if os.path.exists(os.path.join(output_dir, "final", "comm_module.pt")):
                logger.info(f"  [SKIP] {bb}/{cm}: checkpoint already exists at {output_dir}")
                jobs.append({
                    "backbone": bb, "comm_mode": cm, "status": "already_done",
                    "output_dir": output_dir, "has_checkpoint": True,
                    "elapsed_sec": 0, "stderr": "",
                })
                continue
            jobs.append({
                "backbone": bb, "comm_mode": cm, "output_dir": output_dir,
                "trace_files": [trace_files[bb]],
            })

    pending = [j for j in jobs if "status" not in j]
    logger.info(f"  {len(pending)} training jobs to run, {len(jobs) - len(pending)} already done")

    if not pending:
        return jobs

    # Run trainings: group by backbone, run comm modes for each backbone serially
    # (since each backbone uses ~4-8GB VRAM with 4bit quant, we can't fit 2 on one GPU)
    results = []
    for bb in backbones:
        bb_jobs = [j for j in pending if j["backbone"] == bb]
        if not bb_jobs:
            continue

        gpus = GPU_ASSIGNMENTS[bb]
        logger.info(f"\n  Training {bb} ({len(bb_jobs)} jobs on GPUs {gpus})")

        for i, job in enumerate(bb_jobs):
            gpu = gpus[i % len(gpus)]
            res = run_sft_training(
                backbone=job["backbone"],
                comm_mode=job["comm_mode"],
                gpu=gpu,
                trace_files=job["trace_files"],
                output_dir=job["output_dir"],
                quick=quick,
            )
            results.append(res)

    # Merge with already-done jobs
    all_results = [j for j in jobs if j.get("status") == "already_done"] + results
    ok_count = sum(1 for r in all_results if r.get("status") == "ok" or r.get("status") == "already_done")
    logger.info(f"\nPhase B complete: {ok_count}/{len(all_results)} trainings succeeded")

    return all_results


def phase_c_eval_with_checkpoints(
    training_results: List[Dict],
    output_base: str,
) -> None:
    """Phase C: Re-run evaluations using trained checkpoints."""
    logger.info("\n" + "=" * 70)
    logger.info("PHASE C: Evaluation with Trained Checkpoints")
    logger.info("=" * 70)

    successful = [r for r in training_results if r.get("has_checkpoint")]
    if not successful:
        logger.warning("No successful trainings to evaluate!")
        return

    # Generate eval script
    eval_script = os.path.join(output_base, "run_trained_evals.sh")
    with open(eval_script, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Auto-generated: evaluate trained models\n\n")
        for r in successful:
            bb = r["backbone"]
            cm = r["comm_mode"]
            ckpt = os.path.join(r["output_dir"], "final")
            f.write(f"echo 'Evaluating {bb}/{cm}'\n")
            f.write(f"python scripts/run_publication_experiments.py "
                     f"--backbone {bb} --comm_mode {cm} "
                     f"--checkpoint {ckpt} --output_dir {output_base}/eval_{bb}_{cm}\n\n")

    os.chmod(eval_script, 0o755)
    logger.info(f"Eval script written to {eval_script}")
    logger.info("Run it after training completes: bash " + eval_script)


def generate_report(
    training_results: List[Dict],
    output_base: str,
) -> None:
    """Phase D: Generate a summary report."""
    logger.info("\n" + "=" * 70)
    logger.info("PHASE D: Summary Report")
    logger.info("=" * 70)

    report_lines = [
        "=" * 70,
        "SFT TRAINING SUMMARY REPORT",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        "",
    ]

    total_time = sum(r.get("elapsed_sec", 0) for r in training_results)
    n_ok = sum(1 for r in training_results if r.get("status") in ("ok", "already_done"))
    n_total = len(training_results)

    report_lines.append(f"Total trainings: {n_total}")
    report_lines.append(f"Successful: {n_ok}")
    report_lines.append(f"Failed: {n_total - n_ok}")
    report_lines.append(f"Total training time: {total_time/3600:.1f} hours")
    report_lines.append("")
    report_lines.append(f"{'Backbone':<16} {'CommMode':<14} {'Status':<12} {'Time':>10} {'Checkpoint'}")
    report_lines.append("-" * 70)

    for r in training_results:
        bb = r.get("backbone", "?")
        cm = r.get("comm_mode", "?")
        status = r.get("status", "?")
        elapsed = r.get("elapsed_sec", 0)
        ckpt = "✓" if r.get("has_checkpoint") else "✗"
        report_lines.append(f"{bb:<16} {cm:<14} {status:<12} {elapsed/60:>8.1f}m  {ckpt}")

    report_lines.append("")
    report_lines.append("=" * 70)

    report_text = "\n".join(report_lines)
    logger.info("\n" + report_text)

    # Save report
    report_path = os.path.join(output_base, "sft_training_report.txt")
    os.makedirs(output_base, exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report_text)
    logger.info(f"Report saved to {report_path}")

    # Save JSON results
    json_path = os.path.join(output_base, "sft_training_results.json")
    with open(json_path, "w") as f:
        json.dump(training_results, f, indent=2)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Full SFT Pipeline")
    parser.add_argument("--backbones", type=str, nargs="+", default=ALL_BACKBONES)
    parser.add_argument("--comm_modes", type=str, nargs="+", default=ALL_COMM_MODES)
    parser.add_argument("--n_tasks", type=int, default=200,
                        help="Number of tasks per benchmark for trace generation")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 1 epoch, fewer tasks")
    parser.add_argument("--no_model_traces", action="store_true",
                        help="Use GT text only for traces (skip model inference)")
    parser.add_argument("--skip_traces", action="store_true",
                        help="Skip trace generation (use existing)")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    # Quick mode overrides
    n_tasks = 20 if args.quick else args.n_tasks

    # Output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_base = args.output_dir or str(PROJECT_ROOT / "outputs" / f"sft_pipeline_{timestamp}")
    os.makedirs(output_base, exist_ok=True)

    logger.info(f"Output directory: {output_base}")
    logger.info(f"Backbones: {args.backbones}")
    logger.info(f"Comm modes: {args.comm_modes}")
    logger.info(f"Tasks per benchmark: {n_tasks}")

    # Phase A: Generate traces
    if args.skip_traces:
        trace_files = {}
        for bb in args.backbones:
            tf = str(TRACE_DIR / f"traces_{bb}.jsonl")
            if os.path.exists(tf):
                trace_files[bb] = tf
    else:
        trace_files = phase_a_traces(
            backbones=args.backbones,
            n_tasks=n_tasks,
            no_model=args.no_model_traces or args.quick,
        )

    if not trace_files:
        logger.error("No trace files available. Cannot proceed to training.")
        sys.exit(1)

    # Phase B: SFT Training
    training_results = phase_b_training(
        backbones=args.backbones,
        comm_modes=args.comm_modes,
        trace_files=trace_files,
        output_base=output_base,
        quick=args.quick,
    )

    # Phase C: Prepare eval scripts
    phase_c_eval_with_checkpoints(training_results, output_base)

    # Phase D: Report
    generate_report(training_results, output_base)


if __name__ == "__main__":
    main()
