#!/usr/bin/env python3
"""
finetune_benchmarks.py — Fine-tune Latent Agent Team on real benchmark training data.

Fine-tunes LoRA adapters + comm modules on:
  - Mind2Web train split: learn to predict (element, operation, value) from HTML
  - WebLINX train split: learn to predict action from page context
  - AgentInstruct train split: learn conversational agent responses

This is the CRITICAL step that moves us from zero-shot (~5% elem_acc) to SOTA (~40-50%+ elem_acc).

Usage:
    # Single backbone on one GPU
    python scripts/finetune_benchmarks.py --backbone llama32_3b --gpu 0

    # All backbones across 8 GPUs (parallel)
    python scripts/finetune_benchmarks.py --parallel --ngpus 8

    # Only Mind2Web fine-tuning
    python scripts/finetune_benchmarks.py --backbone llama32_3b --benchmarks mind2web --gpu 0
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
import traceback
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("finetune_benchmarks")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

CONFIG_MAP = {
    "phi3_mini": "phi3.yaml",
    "llama32_3b": "llama32_3b.yaml",
    "gemma2_9b": "gemma2_9b.yaml",
    "ministral_8b": "ministral3_8b.yaml",
}


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET CLASSES — Format benchmark data for SFT
# ═══════════════════════════════════════════════════════════════════════════════

class Mind2WebSFTDataset(Dataset):
    """
    Mind2Web SFT dataset: each sample is (HTML_context + instruction) → (operation, element, value).
    
    The model learns to predict the correct element and operation from the page HTML.
    This is the key to high elem_acc scores.
    """
    
    @staticmethod
    def _parse_candidate(cand) -> str:
        """Parse a Mind2Web candidate (pos or neg) into a human-readable element description."""
        try:
            if isinstance(cand, str):
                return cand
            if not isinstance(cand, dict):
                return str(cand)[:100]
            tag = cand.get("tag", "")
            attrs_raw = cand.get("attributes", "")
            # attributes can be a JSON string or a dict
            attrs = {}
            if isinstance(attrs_raw, str):
                try:
                    attrs = json.loads(attrs_raw)
                except (json.JSONDecodeError, TypeError):
                    attrs = {}
            elif isinstance(attrs_raw, dict):
                attrs = attrs_raw
            # Build element description from best identifiers
            label = str(attrs.get("aria-label", "") or "")
            el_id = str(attrs.get("id", "") or "")
            el_class = str(attrs.get("class", "") or "")
            el_name = str(attrs.get("name", "") or "")
            el_text = str(attrs.get("text", cand.get("text", "")) or "")
            el_placeholder = str(attrs.get("placeholder", "") or "")
            # Prioritize: aria-label > text > placeholder > id > name > class
            desc = label or el_text or el_placeholder or el_id or el_name
            if not desc and el_class:
                parts = el_class.strip().split()
                desc = parts[-1] if parts else ""
            if tag:
                desc = f"[{tag}] {desc}" if desc else f"[{tag}]"
            return desc.strip()[:200]
        except Exception:
            return str(cand)[:100]
    
    def __init__(self, tasks: List[Dict], tokenizer, max_seq_len: int = 2048):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        for task in tasks:
            instruction = task["instruction"]
            website = task.get("website", "")
            
            for ai, action in enumerate(task.get("actions", [])):
                op_info = action.get("operation", {})
                op_type = op_info.get("op", "CLICK").upper()
                op_value = op_info.get("value", "")
                
                # Ground truth element from positive candidates
                pos_cands = action.get("pos_candidates", [])
                gt_element = ""
                if pos_cands:
                    # Use the first positive candidate
                    cand = pos_cands[0]
                    gt_element = self._parse_candidate(cand)
                
                # Get action representation
                action_repr = ""
                if ai < len(task.get("action_reprs", [])):
                    action_repr = task["action_reprs"][ai]
                
                html_snippet = str(action.get("cleaned_html", ""))[:1500]
                
                # Build negative candidates for contrastive context
                neg_cands = action.get("neg_candidates", [])[:10]
                neg_texts = []
                for nc in neg_cands:
                    nc_text = self._parse_candidate(nc)
                    if nc_text:
                        neg_texts.append(nc_text[:100])
                
                candidates_text = ""
                if neg_texts or gt_element:
                    all_cands = [("pos", gt_element)] + [("neg", nt) for nt in neg_texts] if gt_element else [("neg", nt) for nt in neg_texts]
                    # Shuffle candidates with deterministic seed
                    import random
                    rng = random.Random(hash((task.get('task_id', ''), ai)))
                    rng.shuffle(all_cands)
                    gt_idx = next((i for i, (kind, _) in enumerate(all_cands) if kind == "pos"), -1)
                    candidates_text = "\nCandidate elements:\n" + "\n".join(f"  [{i+1}] {c}" for i, (_, c) in enumerate(all_cands))
                
                # Input prompt
                input_text = (
                    f"Task: {instruction}\n"
                    f"Website: {website}\n"
                    f"Step {ai+1}/{len(task.get('actions', []))}\n"
                    f"HTML:\n{html_snippet}\n"
                    f"{candidates_text}\n\n"
                    f"Predict the operation (CLICK/TYPE/SELECT), target element, and value.\n"
                    f"Output format: {{\"operation\": \"...\", \"element\": \"...\", \"value\": \"...\"}}"
                )
                
                # Target output: include the candidate index + operation
                target_text = json.dumps({
                    "operation": op_type,
                    "element": str(gt_idx + 1) if gt_idx >= 0 else (gt_element or action_repr),
                    "value": op_value,
                })
                
                self.samples.append({
                    "input": input_text,
                    "target": target_text,
                    "task_id": task.get("task_id", ""),
                    "step": ai,
                })
        
        logger.info(f"Mind2Web SFT dataset: {len(self.samples)} samples from {len(tasks)} tasks")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Tokenize input + target
        full_text = sample["input"] + "\n" + sample["target"]
        input_ids = self.tokenizer(
            full_text, 
            max_length=self.max_seq_len, 
            truncation=True, 
            return_tensors="pt",
            padding=False,
        )["input_ids"].squeeze(0)
        
        # Create labels: mask the input part, only train on target
        input_only = self.tokenizer(
            sample["input"] + "\n", 
            max_length=self.max_seq_len, 
            truncation=True, 
            return_tensors="pt",
            padding=False,
        )["input_ids"].squeeze(0)
        
        labels = input_ids.clone()
        labels[:len(input_only)] = -100  # mask input tokens
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": torch.ones_like(input_ids),
        }


class WebLINXSFTDataset(Dataset):
    """
    WebLINX SFT dataset: given context → predict action.
    """
    
    def __init__(self, samples: List[Dict], tokenizer, max_seq_len: int = 2048):
        self.data = []
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        for s in samples:
            gt_action = s.get("action", "")
            if not gt_action:
                continue
            
            history = (s.get("action_history") or "")[:300]
            utterances = (s.get("utterances") or "")[:300]
            html = (s.get("clean_html") or "")[:1200]
            
            input_text = (
                f"Previous actions: {history}\n"
                f"Utterances: {utterances}\n"
                f"HTML:\n{html}\n\n"
                f"Predict the next action.\n"
                f"Output the action in the exact format used in the dataset."
            )
            
            self.data.append({
                "input": input_text,
                "target": gt_action,
            })
        
        logger.info(f"WebLINX SFT dataset: {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        full_text = sample["input"] + "\n" + sample["target"]
        input_ids = self.tokenizer(
            full_text,
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
            padding=False,
        )["input_ids"].squeeze(0)
        
        input_only = self.tokenizer(
            sample["input"] + "\n",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
            padding=False,
        )["input_ids"].squeeze(0)
        
        labels = input_ids.clone()
        labels[:len(input_only)] = -100
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": torch.ones_like(input_ids),
        }


class AgentInstructSFTDataset(Dataset):
    """
    AgentInstruct SFT dataset: given human instruction → predict assistant response.
    """
    
    def __init__(self, tasks: List[Dict], tokenizer, max_seq_len: int = 2048):
        self.data = []
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        for task in tasks:
            convs = task.get("conversations", [])
            # Build (context, response) pairs from conversation turns
            context = ""
            for turn in convs:
                role = turn.get("from", "")
                content = turn.get("value", "")[:500]
                
                if role == "human":
                    context += f"User: {content}\n"
                elif role == "gpt":
                    if context:
                        self.data.append({
                            "input": context.strip() + "\n\nAssistant response:",
                            "target": content,
                        })
                    context += f"Assistant: {content}\n"
        
        logger.info(f"AgentInstruct SFT dataset: {len(self.data)} samples from {len(tasks)} tasks")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        full_text = sample["input"] + "\n" + sample["target"]
        input_ids = self.tokenizer(
            full_text,
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
            padding=False,
        )["input_ids"].squeeze(0)
        
        input_only = self.tokenizer(
            sample["input"] + "\n",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
            padding=False,
        )["input_ids"].squeeze(0)
        
        labels = input_ids.clone()
        labels[:len(input_only)] = -100
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": torch.ones_like(input_ids),
        }


def collate_fn(batch):
    """Pad batch to same length."""
    max_len = max(b["input_ids"].size(0) for b in batch)
    
    input_ids = torch.full((len(batch), max_len), 0, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    
    for i, b in enumerate(batch):
        seq_len = b["input_ids"].size(0)
        input_ids[i, :seq_len] = b["input_ids"]
        labels[i, :seq_len] = b["labels"]
        attention_mask[i, :seq_len] = b["attention_mask"]
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_mind2web_train(max_tasks: int = 0) -> Tuple[List[Dict], List[Dict]]:
    """Load Mind2Web and split 80/20 for train/test."""
    from datasets import load_dataset
    logger.info("Loading Mind2Web train data...")
    ds = load_dataset("osunlp/Mind2Web", split="train", streaming=True)
    
    all_tasks = []
    for sample in ds:
        task = {
            "task_id": sample["annotation_id"],
            "instruction": sample["confirmed_task"],
            "website": sample["website"],
            "domain": sample["domain"],
            "action_reprs": sample.get("action_reprs", []),
            "actions": [],
        }
        for act in sample.get("actions", []):
            task["actions"].append({
                "action_uid": act.get("action_uid", ""),
                "operation": act.get("operation", {}),
                "cleaned_html": str(act.get("cleaned_html", ""))[:2000],
                "pos_candidates": act.get("pos_candidates", [])[:5],
                "neg_candidates": act.get("neg_candidates", [])[:20],
            })
        all_tasks.append(task)
        if max_tasks and len(all_tasks) >= max_tasks:
            break
    
    # 80/20 split
    split_idx = int(len(all_tasks) * 0.8)
    train_tasks = all_tasks[:split_idx]
    test_tasks = all_tasks[split_idx:]
    logger.info(f"Mind2Web: {len(train_tasks)} train, {len(test_tasks)} test tasks")
    return train_tasks, test_tasks


def load_weblinx_train(max_samples: int = 0) -> List[Dict]:
    """Load WebLINX training data."""
    from datasets import load_dataset
    logger.info("Loading WebLINX train data...")
    ds = load_dataset("McGill-NLP/WebLINX", "chat", split="train", streaming=True)
    samples = []
    for s in ds:
        sample = {
            "demo": s.get("demo", ""),
            "turn": s.get("turn", 0),
            "action": s.get("action") or "",
            "action_history": s.get("action_history") or "",
            "utterances": s.get("utterances") or "",
            "clean_html": (s.get("clean_html") or "")[:3000],
        }
        samples.append(sample)
        if max_samples and len(samples) >= max_samples:
            break
    logger.info(f"WebLINX: {len(samples)} train samples")
    return samples


def load_agentinstruct_train(max_tasks: int = 0) -> List[Dict]:
    """Load AgentInstruct combined training data."""
    from datasets import load_dataset
    logger.info("Loading AgentInstruct train data...")
    all_tasks = []
    for env_type in ["os", "db", "webshop", "alfworld"]:
        try:
            ds = load_dataset("THUDM/AgentInstruct", split=env_type, streaming=True)
            for s in ds:
                convs = s.get("conversations", [])
                task = {
                    "task_id": s.get("id", f"{env_type}_{len(all_tasks)}"),
                    "env_type": env_type,
                    "conversations": convs,
                    "instruction": "",
                }
                for c in convs:
                    if c["from"] == "human":
                        task["instruction"] = c["value"][:500]
                        break
                all_tasks.append(task)
                if max_tasks and len(all_tasks) >= max_tasks:
                    break
        except Exception as e:
            logger.warning(f"Failed to load AgentInstruct/{env_type}: {e}")
        if max_tasks and len(all_tasks) >= max_tasks:
            break
    logger.info(f"AgentInstruct: {len(all_tasks)} train tasks")
    return all_tasks


# ═══════════════════════════════════════════════════════════════════════════════
# FINE-TUNING ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def build_team_for_training(backbone_name: str, comm_mode: str, device: torch.device):
    """Build AgentTeam configured for training (gradient checkpointing on)."""
    from omegaconf import OmegaConf
    from latent_agent_team.team import AgentTeam
    
    cfg_file = PROJECT_ROOT / "configs" / CONFIG_MAP[backbone_name]
    cfg = OmegaConf.load(str(cfg_file))
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict["backbone"]["name"] = backbone_name
    cfg_dict["communication"]["mode"] = comm_mode
    cfg_dict["backbone"]["device_map"] = str(device)
    cfg_dict["backbone"]["gradient_checkpointing"] = True
    
    team = AgentTeam.from_config(cfg_dict, device=device)
    return team


def finetune_on_benchmark(
    team,
    dataset: Dataset,
    benchmark_name: str,
    output_dir: str,
    device: torch.device,
    epochs: int = 3,
    batch_size: int = 2,
    lr: float = 2e-4,
    grad_accum: int = 4,
    max_grad_norm: float = 1.0,
    warmup_ratio: float = 0.1,
):
    """Fine-tune team's LoRA adapters on a benchmark dataset."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Fine-tuning on {benchmark_name}: {len(dataset)} samples, {epochs} epochs")
    logger.info(f"  batch_size={batch_size}, grad_accum={grad_accum}, lr={lr}")
    logger.info(f"  Effective batch size: {batch_size * grad_accum}")
    logger.info(f"{'='*60}")
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0,
        drop_last=True,
    )
    
    # Collect trainable parameters: LoRA + comm module + role adapters
    trainable_params = []
    frozen_count = 0
    
    # LoRA parameters in backbone
    for name, param in team.backbone.model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
            trainable_params.append(param)
        else:
            param.requires_grad = False
            frozen_count += 1
    
    # Comm module parameters
    for param in team.comm.parameters():
        param.requires_grad = True
        trainable_params.append(param)
    
    # Role adapter parameters (summarizers)
    for name, param in team.role_manager.get_all_summarizer_params():
        param.requires_grad = True
        trainable_params.append(param)
    
    # Router + bitrate scheduler
    for param in team.router.parameters():
        param.requires_grad = True
        trainable_params.append(param)
    for param in team.bitrate.parameters():
        param.requires_grad = True
        trainable_params.append(param)
    
    # Audit decoder
    for param in team.audit.parameters():
        param.requires_grad = True
        trainable_params.append(param)
    
    total_trainable = sum(p.numel() for p in trainable_params)
    logger.info(f"  Trainable params: {total_trainable:,} ({frozen_count} frozen backbone params)")
    
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    
    total_steps = (len(dataloader) * epochs) // grad_accum
    warmup_steps = int(total_steps * warmup_ratio)
    
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    # Training loop
    team.backbone.model.train()
    global_step = 0
    best_loss = float("inf")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_steps = 0
        t_epoch = time.time()
        
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Use ONLY the browser adapter for benchmark fine-tuning
            # (the browser agent is the one that produces actions during eval)
            try:
                team.backbone.model.set_adapter("browser")
            except:
                pass
            
            # Forward pass with causal LM loss
            outputs = team.backbone.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            
            loss = outputs.loss / grad_accum
            loss.backward()
            
            epoch_loss += outputs.loss.item()
            epoch_steps += 1
            
            if (batch_idx + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                if global_step % 20 == 0:
                    avg_loss = epoch_loss / epoch_steps
                    lr_now = scheduler.get_last_lr()[0]
                    logger.info(f"  Epoch {epoch+1}/{epochs} step {global_step}/{total_steps}: "
                               f"loss={avg_loss:.4f} lr={lr_now:.2e}")
        
        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        elapsed = time.time() - t_epoch
        logger.info(f"  Epoch {epoch+1}/{epochs} done: avg_loss={avg_epoch_loss:.4f} ({elapsed:.0f}s)")
        
        # Save best
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            save_checkpoint(team, output_dir, benchmark_name)
    
    logger.info(f"Fine-tuning on {benchmark_name} complete. Best loss: {best_loss:.4f}")
    return best_loss


def finetune_all_benchmarks(
    team,
    tokenizer,
    benchmarks: List[str],
    output_dir: str,
    device: torch.device,
    epochs: int = 3,
    max_m2w: int = 0,
    max_wl: int = 0,
    max_ai: int = 0,
):
    """Fine-tune team on all requested benchmarks sequentially."""
    results = {}
    
    if "mind2web" in benchmarks:
        m2w_train, m2w_test = load_mind2web_train(max_tasks=max_m2w)
        ds = Mind2WebSFTDataset(m2w_train, tokenizer, max_seq_len=2048)
        if len(ds) > 0:
            loss = finetune_on_benchmark(
                team, ds, "mind2web", output_dir, device,
                epochs=epochs, batch_size=2, lr=2e-4, grad_accum=8,
            )
            results["mind2web"] = {"train_loss": loss, "train_samples": len(ds)}
            # Save test split for later evaluation
            test_file = os.path.join(output_dir, "mind2web_test_tasks.json")
            with open(test_file, "w") as f:
                json.dump(m2w_test, f)
        del ds, m2w_train
        gc.collect(); torch.cuda.empty_cache()
    
    if "weblinx" in benchmarks:
        wl_train = load_weblinx_train(max_samples=max_wl)
        ds = WebLINXSFTDataset(wl_train, tokenizer, max_seq_len=2048)
        if len(ds) > 0:
            loss = finetune_on_benchmark(
                team, ds, "weblinx", output_dir, device,
                epochs=epochs, batch_size=2, lr=2e-4, grad_accum=8,
            )
            results["weblinx"] = {"train_loss": loss, "train_samples": len(ds)}
        del ds, wl_train
        gc.collect(); torch.cuda.empty_cache()
    
    if "agentinstruct" in benchmarks:
        ai_train = load_agentinstruct_train(max_tasks=max_ai)
        ds = AgentInstructSFTDataset(ai_train, tokenizer, max_seq_len=2048)
        if len(ds) > 0:
            loss = finetune_on_benchmark(
                team, ds, "agentinstruct", output_dir, device,
                epochs=epochs, batch_size=2, lr=2e-4, grad_accum=4,
            )
            results["agentinstruct"] = {"train_loss": loss, "train_samples": len(ds)}
        del ds, ai_train
        gc.collect(); torch.cuda.empty_cache()
    
    return results


def save_checkpoint(team, output_dir: str, benchmark_name: str):
    """Save fine-tuned checkpoint."""
    ckpt_dir = os.path.join(output_dir, f"checkpoint_{benchmark_name}")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Save LoRA adapters
    lora_dir = os.path.join(ckpt_dir, "lora_adapters")
    os.makedirs(lora_dir, exist_ok=True)
    for role_name in ["planner", "retriever", "browser", "verifier", "memory"]:
        try:
            role_dir = os.path.join(lora_dir, role_name, role_name)
            os.makedirs(role_dir, exist_ok=True)
            team.backbone.model.set_adapter(role_name)
            team.backbone.model.save_pretrained(
                os.path.join(lora_dir, role_name),
                selected_adapters=[role_name],
            )
        except Exception as e:
            logger.debug(f"Could not save LoRA for {role_name}: {e}")
    
    # Save comm module
    torch.save(team.comm.state_dict(), os.path.join(ckpt_dir, "comm_module.pt"))
    
    # Save role adapters (summarizers)
    for role_name in ["planner", "retriever", "browser", "verifier", "memory"]:
        try:
            adapter = team.role_manager.get_adapter(role_name)
            torch.save(
                adapter.summarizer.state_dict(),
                os.path.join(ckpt_dir, f"summarizer_{role_name}.pt"),
            )
        except Exception as e:
            logger.debug(f"Could not save summarizer for {role_name}: {e}")
    
    # Save router, bitrate, audit
    torch.save(team.router.state_dict(), os.path.join(ckpt_dir, "sparse_router.pt"))
    torch.save(team.bitrate.state_dict(), os.path.join(ckpt_dir, "bitrate_scheduler.pt"))
    torch.save(team.audit.state_dict(), os.path.join(ckpt_dir, "audit_decoder.pt"))
    
    logger.info(f"Checkpoint saved to {ckpt_dir}")


def save_final_merged_checkpoint(team, output_dir: str):
    """Save a final merged checkpoint combining all benchmark fine-tuning."""
    final_dir = os.path.join(output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    
    # Save LoRA adapters
    lora_dir = os.path.join(final_dir, "lora_adapters")
    os.makedirs(lora_dir, exist_ok=True)
    for role_name in ["planner", "retriever", "browser", "verifier", "memory"]:
        try:
            role_dir = os.path.join(lora_dir, role_name, role_name)
            os.makedirs(role_dir, exist_ok=True)
            team.backbone.model.set_adapter(role_name)
            team.backbone.model.save_pretrained(
                os.path.join(lora_dir, role_name),
                selected_adapters=[role_name],
            )
        except Exception as e:
            logger.debug(f"Could not save LoRA for {role_name}: {e}")
    
    # Save all modules
    torch.save(team.comm.state_dict(), os.path.join(final_dir, "comm_module.pt"))
    torch.save(team.router.state_dict(), os.path.join(final_dir, "sparse_router.pt"))
    torch.save(team.bitrate.state_dict(), os.path.join(final_dir, "bitrate_scheduler.pt"))
    torch.save(team.audit.state_dict(), os.path.join(final_dir, "audit_decoder.pt"))
    
    for role_name in ["planner", "retriever", "browser", "verifier", "memory"]:
        try:
            adapter = team.role_manager.get_adapter(role_name)
            torch.save(
                adapter.summarizer.state_dict(),
                os.path.join(final_dir, f"summarizer_{role_name}.pt"),
            )
        except:
            pass
    
    logger.info(f"Final merged checkpoint saved to {final_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="llama32_3b", choices=list(CONFIG_MAP.keys()))
    parser.add_argument("--comm_mode", default="text")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--benchmarks", nargs="+", default=["mind2web", "weblinx", "agentinstruct"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--sft_dir", default="outputs/sft_all_20260309_110924",
                        help="Pre-trained SFT checkpoint dir (load before benchmark FT)")
    parser.add_argument("--max_m2w", type=int, default=0, help="Max Mind2Web tasks (0=all)")
    parser.add_argument("--max_wl", type=int, default=0, help="Max WebLINX samples (0=all ~5000)")
    parser.add_argument("--max_ai", type=int, default=0, help="Max AgentInstruct tasks (0=all ~1700)")
    parser.add_argument("--parallel", action="store_true", help="Launch all backbones across GPUs")
    parser.add_argument("--ngpus", type=int, default=8)
    args = parser.parse_args()
    
    project_root = str(PROJECT_ROOT)
    if not os.path.isabs(args.sft_dir):
        args.sft_dir = os.path.join(project_root, args.sft_dir)
    
    # ── PARALLEL MODE ──
    if args.parallel:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        shared_output = args.output_dir or os.path.join(project_root, f"outputs/benchmark_ft_{timestamp}")
        os.makedirs(shared_output, exist_ok=True)
        
        # Build work units: all backbone × comm_mode combos
        backbones = list(CONFIG_MAP.keys())
        comm_modes = ["continuous", "vq", "text"]
        work_units = [(bb, cm) for bb in backbones for cm in comm_modes]
        
        logger.info(f"PARALLEL: {len(work_units)} fine-tuning jobs across {args.ngpus} GPUs")
        
        all_procs = []
        for batch_start in range(0, len(work_units), args.ngpus):
            batch = work_units[batch_start:batch_start + args.ngpus]
            batch_num = batch_start // args.ngpus + 1
            total_batches = (len(work_units) + args.ngpus - 1) // args.ngpus
            logger.info(f"\n--- Batch {batch_num}/{total_batches} ---")
            
            procs = []
            for j, (bb, cm) in enumerate(batch):
                gpu = j
                unit_output = os.path.join(shared_output, f"ft_{bb}_{cm}")
                cmd = [
                    sys.executable, os.path.abspath(__file__),
                    "--backbone", bb,
                    "--comm_mode", cm,
                    "--gpu", "0",
                    "--benchmarks"] + args.benchmarks + [
                    "--epochs", str(args.epochs),
                    "--output_dir", unit_output,
                    "--sft_dir", args.sft_dir,
                    "--max_m2w", str(args.max_m2w),
                    "--max_wl", str(args.max_wl),
                    "--max_ai", str(args.max_ai),
                ]
                
                log_file = os.path.join(shared_output, f"log_ft_{bb}_{cm}.txt")
                logger.info(f"  GPU {gpu}: {bb}/{cm}")
                with open(log_file, "w") as lf:
                    p = subprocess.Popen(
                        cmd, stdout=lf, stderr=subprocess.STDOUT,
                        env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
                    )
                    procs.append((p, bb, cm))
                time.sleep(3)
            
            # Wait for batch
            for p, bb, cm in procs:
                p.wait()
                status = "✓" if p.returncode == 0 else f"✗ (rc={p.returncode})"
                logger.info(f"  {status} {bb}/{cm}")
                all_procs.append((p.returncode, bb, cm))
        
        # Summary
        n_ok = sum(1 for rc, _, _ in all_procs if rc == 0)
        logger.info(f"\nPARALLEL DONE: {n_ok}/{len(all_procs)} succeeded")
        logger.info(f"Checkpoints in: {shared_output}")
        return
    
    # ── SINGLE MODE ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join(
        project_root, f"outputs/benchmark_ft_{args.backbone}_{args.comm_mode}_{timestamp}"
    )
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Backbone: {args.backbone}, Comm: {args.comm_mode}")
    
    # Build team
    team = build_team_for_training(args.backbone, args.comm_mode, device)
    
    # Load pre-trained SFT checkpoint
    sft_ckpt = os.path.join(args.sft_dir, f"sft_{args.backbone}_{args.comm_mode}", "final")
    if os.path.exists(sft_ckpt):
        logger.info(f"Loading pre-trained SFT from {sft_ckpt}")
        # Reuse the checkpoint loading from run_real_experiments
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from run_real_experiments import load_checkpoint
        load_checkpoint(team, args.sft_dir, args.backbone, args.comm_mode)
    else:
        logger.warning(f"No pre-trained SFT checkpoint at {sft_ckpt}")
    
    # Fine-tune on benchmarks
    results = finetune_all_benchmarks(
        team=team,
        tokenizer=team.backbone.tokenizer,
        benchmarks=args.benchmarks,
        output_dir=output_dir,
        device=device,
        epochs=args.epochs,
        max_m2w=args.max_m2w,
        max_wl=args.max_wl,
        max_ai=args.max_ai,
    )
    
    # Save final merged checkpoint
    save_final_merged_checkpoint(team, output_dir)
    
    # Save training results
    with open(os.path.join(output_dir, "training_results.json"), "w") as f:
        json.dump({
            "backbone": args.backbone,
            "comm_mode": args.comm_mode,
            "benchmarks": args.benchmarks,
            "epochs": args.epochs,
            "results": results,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"FINE-TUNING COMPLETE: {args.backbone}/{args.comm_mode}")
    logger.info(f"Results: {json.dumps(results, indent=2)}")
    logger.info(f"Checkpoint: {output_dir}/final/")
    logger.info(f"{'='*60}")
    
    # Cleanup
    del team
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
