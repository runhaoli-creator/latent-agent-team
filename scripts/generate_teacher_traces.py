#!/usr/bin/env python3
"""
generate_teacher_traces.py — Generate teacher traces for SFT training.

Uses the pretrained backbone (no LoRA) in text-mode to produce supervised
trajectories across all 3 benchmarks. These traces provide the ground-truth
signal for training the latent communication modules.

Each trace step records:
  - The observation the agent saw
  - The text message the agent produced
  - The action taken
  - Which agents received the message (routing targets)
  - The verifier judgment

Output: JSONL files in data/traces/ matching the TraceStep format expected
by TeacherTraceDataset in sft_bootstrap.py.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

from latent_agent_team.models.backbone import BackboneManager, BackboneConfig, BACKBONE_REGISTRY

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("trace_gen")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TRACE_DIR = DATA_DIR / "traces"

# ── Role prompts for trace generation ──────────────────────────────────────────

ROLE_PROMPTS = {
    "planner": (
        "You are a Planner agent. Given a task and observation, produce a JSON action:\n"
        '{"action": "plan", "sub_goals": ["<goal1>", "<goal2>"], "next_agent": "browser", "reasoning": "..."}\n'
    ),
    "retriever": (
        "You are a Retriever agent. Given a query, find relevant information:\n"
        '{"action": "retrieve", "query": "...", "results": ["<evidence1>"], "confidence": 0.8}\n'
    ),
    "browser": (
        "You are a Browser agent interacting with a web page. Produce a JSON action:\n"
        '{"action": "click|type|search|select", "element": "...", "value": "...", "reasoning": "..."}\n'
    ),
    "verifier": (
        "You are a Verifier agent. Check if the action matches the task goal:\n"
        '{"action": "verify", "status": "pass|fail|uncertain", "issues": [], "suggestion": "..."}\n'
    ),
}

# ── Benchmark task generators ──────────────────────────────────────────────────

def make_webshop_trace_tasks(n: int = 200) -> List[Dict]:
    """Generate WebShop tasks with ground-truth action sequences."""
    import random
    rng = random.Random(42)

    items = [
        {"name": "wireless headphones", "color": "blue", "price": 29.99, "attrs": {"color": "blue", "type": "wireless"}},
        {"name": "running shoes", "color": "red", "price": 59.99, "attrs": {"color": "red", "size": "10"}},
        {"name": "laptop bag", "color": "black", "price": 35.00, "attrs": {"color": "black", "type": "laptop"}},
        {"name": "water bottle", "color": "green", "price": 15.99, "attrs": {"color": "green", "size": "32oz"}},
        {"name": "phone case", "color": "clear", "price": 12.99, "attrs": {"color": "clear", "type": "protective"}},
        {"name": "desk lamp", "color": "white", "price": 25.99, "attrs": {"color": "white", "type": "LED"}},
        {"name": "keyboard", "color": "black", "price": 49.99, "attrs": {"color": "black", "type": "mechanical"}},
        {"name": "backpack", "color": "navy", "price": 44.99, "attrs": {"color": "navy", "size": "25L"}},
        {"name": "yoga mat", "color": "purple", "price": 22.99, "attrs": {"color": "purple", "thickness": "6mm"}},
        {"name": "sunglasses", "color": "black", "price": 18.99, "attrs": {"color": "black", "type": "polarized"}},
        {"name": "t-shirt", "color": "white", "price": 14.99, "attrs": {"color": "white", "size": "M"}},
        {"name": "notebook", "color": "brown", "price": 8.99, "attrs": {"color": "brown", "pages": "200"}},
    ]

    tasks = []
    for i in range(n):
        item = rng.choice(items)
        instruction = f"Find a {item['color']} {item['name']} under ${item['price'] + 20:.0f}"
        # Ground-truth action sequence
        gt_actions = [
            {"role": "planner", "action": json.dumps({"action": "plan", "sub_goals": [f"Search for {item['color']} {item['name']}", "Select matching product", "Buy it"], "next_agent": "browser", "reasoning": f"Need to find {item['color']} {item['name']}"}),
             "message": f"Search for {item['color']} {item['name']} and buy the best match under ${item['price'] + 20:.0f}",
             "recipients": ["browser", "retriever"]},
            {"role": "retriever", "action": json.dumps({"action": "retrieve", "query": f"{item['color']} {item['name']}", "results": [f"{item['name']} - ${item['price']:.2f}"], "confidence": 0.85}),
             "message": f"Found {item['name']} at ${item['price']:.2f} matching criteria",
             "recipients": ["browser"]},
            {"role": "browser", "action": json.dumps({"action": "search", "element": "search_bar", "value": f"{item['color']} {item['name']}", "reasoning": "Searching for the product"}),
             "message": f"Searching for {item['color']} {item['name']}",
             "recipients": ["verifier"]},
            {"role": "browser", "action": json.dumps({"action": "click", "element": item['name'], "value": "", "reasoning": f"Selecting {item['name']} from results"}),
             "message": f"Found and clicking on {item['name']}",
             "recipients": ["verifier"]},
            {"role": "verifier", "action": json.dumps({"action": "verify", "status": "pass", "issues": [], "suggestion": "Product matches criteria, proceed to buy"}),
             "message": "Product verified, matches requirements",
             "recipients": ["planner", "browser"]},
            {"role": "browser", "action": json.dumps({"action": "click", "element": "Buy Now", "value": "", "reasoning": "Completing purchase"}),
             "message": "Clicking Buy Now to complete purchase",
             "recipients": ["verifier"]},
        ]
        tasks.append({
            "task_id": f"webshop_{i:04d}",
            "instruction": instruction,
            "gt_actions": gt_actions,
            "item": item,
        })
    return tasks


def make_mind2web_trace_tasks(n: int = 200) -> List[Dict]:
    """Generate Mind2Web tasks with ground-truth action sequences."""
    import random
    rng = random.Random(43)

    websites = [
        {"site": "google.com", "tasks": [
            ("Search for machine learning papers", [
                {"op": "click", "element": "search_bar", "value": ""},
                {"op": "type", "element": "search_bar", "value": "machine learning papers 2026"},
                {"op": "click", "element": "search_button", "value": ""},
            ]),
            ("Navigate to Google Scholar", [
                {"op": "click", "element": "menu_button", "value": ""},
                {"op": "click", "element": "Scholar link", "value": ""},
            ]),
        ]},
        {"site": "amazon.com", "tasks": [
            ("Find wireless earbuds under $50", [
                {"op": "type", "element": "search_bar", "value": "wireless earbuds"},
                {"op": "click", "element": "search_button", "value": ""},
                {"op": "click", "element": "price_filter_under_50", "value": ""},
            ]),
            ("Add item to cart", [
                {"op": "click", "element": "first_result", "value": ""},
                {"op": "click", "element": "add_to_cart", "value": ""},
            ]),
        ]},
        {"site": "wikipedia.org", "tasks": [
            ("Find article on neural networks", [
                {"op": "type", "element": "search_input", "value": "neural network"},
                {"op": "click", "element": "search_button", "value": ""},
            ]),
            ("Navigate to References section", [
                {"op": "click", "element": "toc_references", "value": ""},
            ]),
        ]},
        {"site": "github.com", "tasks": [
            ("Search for transformer repositories", [
                {"op": "type", "element": "search_input", "value": "transformer pytorch"},
                {"op": "click", "element": "search_button", "value": ""},
                {"op": "click", "element": "first_repo", "value": ""},
            ]),
        ]},
        {"site": "booking.com", "tasks": [
            ("Search for hotels in Paris", [
                {"op": "type", "element": "destination_input", "value": "Paris, France"},
                {"op": "click", "element": "date_checkin", "value": "2026-04-01"},
                {"op": "click", "element": "date_checkout", "value": "2026-04-05"},
                {"op": "click", "element": "search_button", "value": ""},
            ]),
        ]},
    ]

    tasks = []
    for i in range(n):
        site_data = rng.choice(websites)
        task_name, dom_actions = rng.choice(site_data["tasks"])
        instruction = f"{task_name} on {site_data['site']}"

        gt_actions = []
        for step_j, dom_act in enumerate(dom_actions):
            gt_actions.append({
                "role": "planner" if step_j == 0 else "browser",
                "action": json.dumps({"action": dom_act["op"], "element": dom_act["element"], "value": dom_act["value"], "reasoning": f"Step {step_j+1}: {dom_act['op']} on {dom_act['element']}"}),
                "message": f"Performing {dom_act['op']} on {dom_act['element']}" + (f" with value '{dom_act['value']}'" if dom_act['value'] else ""),
                "recipients": ["browser", "verifier"] if step_j == 0 else ["verifier"],
                "dom_html": f"<div id='page'><input id='{dom_act['element']}' type='text'><button id='submit'>Go</button></div>",
                "gt_op": dom_act["op"],
                "gt_value": dom_act["value"],
            })
        # Add verifier step
        gt_actions.append({
            "role": "verifier",
            "action": json.dumps({"action": "verify", "status": "pass", "issues": [], "suggestion": "Task completed successfully"}),
            "message": "All steps verified successfully",
            "recipients": ["planner"],
        })

        tasks.append({
            "task_id": f"mind2web_{i:04d}",
            "instruction": instruction,
            "website": site_data["site"],
            "gt_actions": gt_actions,
        })
    return tasks


def make_agentbench_trace_tasks(n: int = 100) -> List[Dict]:
    """Generate AgentBench tasks with ground-truth action sequences."""
    import random
    rng = random.Random(44)

    os_tasks = [
        {"instruction": "Count the number of .txt files in /tmp",
         "commands": ["find /tmp -name '*.txt' | wc -l"], "expected": "0"},
        {"instruction": "List all files in /etc sorted by size",
         "commands": ["ls -lS /etc | head -20"], "expected": ""},
        {"instruction": "Find the system hostname",
         "commands": ["cat /etc/hostname"], "expected": ""},
        {"instruction": "Check disk usage of /tmp",
         "commands": ["du -sh /tmp"], "expected": ""},
        {"instruction": "Find all Python processes",
         "commands": ["ps aux | grep python"], "expected": "python"},
        {"instruction": "Count lines in /etc/passwd",
         "commands": ["wc -l /etc/passwd"], "expected": ""},
        {"instruction": "Show current working directory",
         "commands": ["pwd"], "expected": "/tmp"},
        {"instruction": "List environment variables containing HOME",
         "commands": ["env | grep HOME"], "expected": "HOME"},
    ]

    db_tasks = [
        {"instruction": "Find the employee with the highest salary",
         "commands": ["SELECT name FROM employees ORDER BY salary DESC LIMIT 1"], "expected": "Carol"},
        {"instruction": "Count employees in Engineering",
         "commands": ["SELECT COUNT(*) FROM employees WHERE department='Engineering'"], "expected": "2"},
        {"instruction": "Calculate average salary",
         "commands": ["SELECT AVG(salary) FROM employees"], "expected": ""},
        {"instruction": "List departments with total salary",
         "commands": ["SELECT department, SUM(salary) FROM employees GROUP BY department"], "expected": ""},
        {"instruction": "Find employees earning above 80000",
         "commands": ["SELECT name, salary FROM employees WHERE salary > 80000"], "expected": ""},
    ]

    tasks = []
    for i in range(n):
        if rng.random() < 0.6:
            task_template = rng.choice(os_tasks)
            env_type = "os"
        else:
            task_template = rng.choice(db_tasks)
            env_type = "db"

        gt_actions = [
            {"role": "planner",
             "action": json.dumps({"action": "plan", "sub_goals": [f"Execute: {task_template['commands'][0]}"], "next_agent": "browser", "reasoning": f"Need to {task_template['instruction'].lower()}"}),
             "message": f"Plan: execute command to {task_template['instruction'].lower()}",
             "recipients": ["browser", "retriever"]},
        ]
        for cmd in task_template["commands"]:
            gt_actions.append({
                "role": "browser",
                "action": json.dumps({"action": "type", "element": "terminal", "value": cmd, "reasoning": f"Executing: {cmd}"}),
                "message": f"Executing command: {cmd}",
                "recipients": ["verifier"],
            })
        gt_actions.append({
            "role": "verifier",
            "action": json.dumps({"action": "verify", "status": "pass", "issues": [], "suggestion": "Command executed successfully"}),
            "message": "Task completed, output matches expected",
            "recipients": ["planner"],
        })

        tasks.append({
            "task_id": f"agentbench_{env_type}_{i:04d}",
            "instruction": task_template["instruction"],
            "env_type": env_type,
            "gt_actions": gt_actions,
            "expected": task_template["expected"],
        })
    return tasks


# ── Trace generation with real model ────────────────────────────────────────────

def generate_model_response(
    backbone: BackboneManager,
    role: str,
    observation: str,
    task_instruction: str,
    device: torch.device,
    max_new_tokens: int = 128,
) -> str:
    """Use the real backbone to generate a response for the given role."""
    prompt = (
        f"{ROLE_PROMPTS.get(role, '')}\n"
        f"[TASK]: {task_instruction[:200]}\n"
        f"[OBSERVATION]: {observation[:400]}\n"
        f"[ACTION]:"
    )
    tokenizer = backbone.tokenizer
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    backbone.set_active_adapter(role)
    with torch.no_grad():
        out = backbone.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    new_ids = out[:, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids[0], skip_special_tokens=True).strip()


def generate_traces_for_backbone(
    backbone_name: str,
    device: torch.device,
    n_webshop: int = 200,
    n_mind2web: int = 200,
    n_agentbench: int = 100,
    use_model_responses: bool = True,
) -> List[Dict]:
    """
    Generate teacher traces for a specific backbone.
    
    Uses ground-truth action sequences as the supervision signal,
    augmented with real model-generated text messages for naturalness.
    """
    logger.info(f"Generating traces for {backbone_name} on {device}")

    # Load backbone
    backbone_cfg = BackboneConfig(
        backbone_name=backbone_name,
        quantization="4bit",
        device_map=str(device),
    )
    backbone = BackboneManager(backbone_cfg)
    for role in ["planner", "retriever", "browser", "verifier", "memory"]:
        backbone.add_adapter(role)

    all_traces = []

    # ── WebShop traces ────────────────────────────────────────────────
    logger.info(f"  Generating {n_webshop} WebShop traces...")
    webshop_tasks = make_webshop_trace_tasks(n_webshop)
    for task in webshop_tasks:
        episode_id = f"ws_{task['task_id']}_{uuid.uuid4().hex[:6]}"
        for step_idx, gt_act in enumerate(task["gt_actions"]):
            role = gt_act["role"]
            obs = f"WebShop page for: {task['instruction']}"

            # Use model to generate natural text message, or use GT
            if use_model_responses:
                try:
                    model_msg = generate_model_response(
                        backbone, role, obs, task["instruction"], device, max_new_tokens=64
                    )
                except Exception:
                    model_msg = gt_act["message"]
            else:
                model_msg = gt_act["message"]

            trace = {
                "step_id": step_idx,
                "episode_id": episode_id,
                "sender_role": role,
                "observation": obs,
                "outgoing_text_message": model_msg[:200],
                "next_action": gt_act["action"],
                "selected_evidence": f"Product: {task['item']['name']}" if "item" in task else "",
                "verifier_label": "pass" if role == "verifier" else "uncertain",
                "recipient_roles": gt_act.get("recipients", ["browser"]),
                "benchmark": "webshop",
                "task_instruction": task["instruction"],
            }
            all_traces.append(trace)

    # ── Mind2Web traces ───────────────────────────────────────────────
    logger.info(f"  Generating {n_mind2web} Mind2Web traces...")
    m2w_tasks = make_mind2web_trace_tasks(n_mind2web)
    for task in m2w_tasks:
        episode_id = f"m2w_{task['task_id']}_{uuid.uuid4().hex[:6]}"
        for step_idx, gt_act in enumerate(task["gt_actions"]):
            role = gt_act["role"]
            obs = gt_act.get("dom_html", f"Page on {task.get('website', 'unknown')}")

            if use_model_responses:
                try:
                    model_msg = generate_model_response(
                        backbone, role, obs, task["instruction"], device, max_new_tokens=64
                    )
                except Exception:
                    model_msg = gt_act["message"]
            else:
                model_msg = gt_act["message"]

            trace = {
                "step_id": step_idx,
                "episode_id": episode_id,
                "sender_role": role,
                "observation": obs,
                "outgoing_text_message": model_msg[:200],
                "next_action": gt_act["action"],
                "selected_evidence": gt_act.get("dom_html", ""),
                "verifier_label": "pass" if role == "verifier" else "uncertain",
                "recipient_roles": gt_act.get("recipients", ["browser"]),
                "benchmark": "mind2web",
                "task_instruction": task["instruction"],
            }
            all_traces.append(trace)

    # ── AgentBench traces ─────────────────────────────────────────────
    logger.info(f"  Generating {n_agentbench} AgentBench traces...")
    ab_tasks = make_agentbench_trace_tasks(n_agentbench)
    for task in ab_tasks:
        episode_id = f"ab_{task['task_id']}_{uuid.uuid4().hex[:6]}"
        for step_idx, gt_act in enumerate(task["gt_actions"]):
            role = gt_act["role"]
            obs = f"Terminal: {task.get('env_type', 'os')} environment. Task: {task['instruction']}"

            if use_model_responses:
                try:
                    model_msg = generate_model_response(
                        backbone, role, obs, task["instruction"], device, max_new_tokens=64
                    )
                except Exception:
                    model_msg = gt_act["message"]
            else:
                model_msg = gt_act["message"]

            trace = {
                "step_id": step_idx,
                "episode_id": episode_id,
                "sender_role": role,
                "observation": obs,
                "outgoing_text_message": model_msg[:200],
                "next_action": gt_act["action"],
                "selected_evidence": "",
                "verifier_label": "pass" if role == "verifier" else "uncertain",
                "recipient_roles": gt_act.get("recipients", ["browser"]),
                "benchmark": "agentbench",
                "task_instruction": task["instruction"],
            }
            all_traces.append(trace)

    # Cleanup
    del backbone
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    logger.info(f"  Generated {len(all_traces)} trace steps for {backbone_name}")
    return all_traces


def save_traces(traces: List[Dict], output_path: str) -> None:
    """Save traces to JSONL file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for trace in traces:
            f.write(json.dumps(trace) + "\n")
    logger.info(f"Saved {len(traces)} traces to {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate teacher traces")
    parser.add_argument("--backbone", type=str, default="all",
                        help="Backbone name or 'all'")
    parser.add_argument("--gpu", type=int, default=0, help="GPU to use")
    parser.add_argument("--n_webshop", type=int, default=200)
    parser.add_argument("--n_mind2web", type=int, default=200)
    parser.add_argument("--n_agentbench", type=int, default=100)
    parser.add_argument("--no_model", action="store_true",
                        help="Skip model responses, use GT text only (faster)")
    parser.add_argument("--output_dir", type=str, default=str(TRACE_DIR))
    args = parser.parse_args()

    TRACE_DIR.mkdir(parents=True, exist_ok=True)

    backbones = BACKBONE_REGISTRY.keys() if args.backbone == "all" else [args.backbone]

    for bb_name in backbones:
        device = torch.device(f"cuda:{args.gpu}")
        logger.info(f"\n{'='*60}")
        logger.info(f"Generating traces for {bb_name}")
        logger.info(f"{'='*60}")

        traces = generate_traces_for_backbone(
            backbone_name=bb_name,
            device=device,
            n_webshop=args.n_webshop,
            n_mind2web=args.n_mind2web,
            n_agentbench=args.n_agentbench,
            use_model_responses=not args.no_model,
        )

        output_path = os.path.join(args.output_dir, f"traces_{bb_name}.jsonl")
        save_traces(traces, output_path)

    logger.info("\nAll traces generated!")


if __name__ == "__main__":
    main()
