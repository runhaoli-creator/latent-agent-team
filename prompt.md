# Critical Ablation: Single-Agent Browser Baseline

## Why This Experiment Matters

The paper "Latent Agent Team" claims that latent communication between 5 agents is the key contribution. But the ablation table (Table 5 in the paper) shows:

- SFT-only with full latent communication: **0.4% ElemAcc**
- After Stage 3 browser-focused FT: **81.5% ElemAcc**

Stage 3 browser FT **freezes** all communication modules and only trains the browser LoRA adapter. This means the 80+ pp jump has nothing to do with latent communication.

**The critical missing experiment**: does a plain single-agent browser (no multi-agent team, no latent communication, no SFT Stage 1/2) with the same Stage 3 fine-tuning achieve similar results?

- If single-agent gets ~78%+: the latent communication framework is unnecessary, the paper's story collapses.
- If single-agent gets ~65-70%: latent communication contributes meaningfully via the SFT-pretrained adapter weights.
- If single-agent gets <60%: the multi-agent SFT pretraining is essential, the paper's story is strong.

## What to Build

A **single-agent browser baseline** script: `scripts/ablation_single_agent.py`

This script does:
1. Load a Phi-3 Mini backbone with 4-bit NF4 quantization (same as full system)
2. Add ONE LoRA adapter (r=16, alpha=32) — the "browser" adapter
3. Fine-tune it directly on Mind2Web train data using the **exact same** dataset class and training procedure as Stage 3 in `scripts/finetune_benchmarks.py`
4. Evaluate on Mind2Web test (200 samples) using the **exact same** evaluation code

**No multi-agent team. No latent communication. No SFT pretraining. No DPO. Just a single LoRA adapter trained on benchmark data.**

## Existing Code to Reuse

All the building blocks already exist in the repo:

### Backbone loading
```python
# From src/latent_agent_team/models/backbone.py
from latent_agent_team.models.backbone import BackboneManager, BackboneConfig

cfg = BackboneConfig(
    backbone_name="phi3_mini",
    quantization="4bit",
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    max_seq_len=4096,
    dtype="bfloat16",
    gradient_checkpointing=True,
)
backbone = BackboneManager(cfg)
backbone.add_adapter("browser")  # Only one adapter
backbone.set_active_adapter("browser")
```

### Dataset
Reuse `Mind2WebSFTDataset` from `scripts/finetune_benchmarks.py` exactly as-is. It already:
- Parses candidates from Mind2Web HuggingFace dataset
- Shuffles candidates with deterministic seeds (critical for fair comparison)
- Formats input as: task + HTML + shuffled candidate list → predict operation + element index + value
- Masks input tokens in labels (only trains on target)

### Training loop
Reuse the training procedure from `finetune_on_benchmark()` in `scripts/finetune_benchmarks.py`:
- AdamW, lr=2e-4, weight_decay=0.01
- Cosine schedule with 10% warmup
- 3 epochs, batch_size=2, grad_accum=4 (effective batch=8)
- max_grad_norm=1.0
- Only difference: no comm module / router / bitrate / audit params — just LoRA params

### Evaluation
The evaluation needs to match what the paper reports. The eval code from `scripts/eval_finetuned.py` imports from `run_real_experiments.py` which is missing from the repo. You'll need to implement Mind2Web evaluation yourself:

**Mind2Web evaluation protocol:**
1. Load test split (200 samples) from HuggingFace `osunlp/Mind2Web`
2. For each action step in each task:
   - Build the same input prompt as training (task + HTML + shuffled candidates)
   - Generate model output (greedy, no sampling)
   - Parse the predicted element index and operation from JSON output
   - Compare against ground truth:
     - **ElemAcc**: predicted element index == ground truth element index (after shuffling)
     - **OpF1**: predicted operation type matches ground truth operation type
     - **StepSR**: both element and operation correct
3. Report aggregate metrics

## Implementation Plan

### Step 1: Create `scripts/ablation_single_agent.py`

```python
"""
Single-agent browser baseline for ablation study.
Tests whether a plain browser LoRA adapter with benchmark FT
achieves comparable results to the full Latent Agent Team.

Usage:
    python scripts/ablation_single_agent.py --gpu 0
    python scripts/ablation_single_agent.py --gpu 0 --max_train_tasks 500 --max_test_tasks 200
"""
```

The script should:
1. Parse args: `--gpu`, `--max_train_tasks` (default 0 = all), `--max_test_tasks` (default 200), `--epochs` (default 3), `--batch_size` (default 2), `--lr` (default 2e-4), `--output_dir` (default `outputs/ablation_single_agent`)
2. Load backbone (Phi-3 Mini, 4-bit, single "browser" LoRA adapter)
3. Load Mind2Web train data using `load_mind2web_train()` from `finetune_benchmarks.py`
4. Create `Mind2WebSFTDataset` from `finetune_benchmarks.py`
5. Train with the same loop as `finetune_on_benchmark()`, but ONLY LoRA params (no comm/router/etc)
6. Evaluate on Mind2Web test split
7. Save results as JSON

### Step 2: Implement evaluation

Since `run_real_experiments.py` is missing, implement `evaluate_mind2web_single_agent()`:

```python
@torch.no_grad()
def evaluate_mind2web_single_agent(model, tokenizer, test_tasks, max_tasks=200, device=None):
    """
    Evaluate a single model (no team) on Mind2Web.
    Returns dict with elem_acc, op_f1, step_sr.
    """
    # For each task's each action step:
    # 1. Build the same prompt as Mind2WebSFTDataset (shuffled candidates)
    # 2. Generate with model.generate(greedy)
    # 3. Parse JSON output → extract element index and operation
    # 4. Compare against ground truth
```

Key details:
- Use `Mind2WebSFTDataset._parse_candidate()` for candidate formatting
- Shuffle candidates with the same deterministic seed: `random.Random(hash((task_id, action_idx)))`
- Generate with: `max_new_tokens=128, do_sample=False, temperature=1.0`
- Parse output JSON to get predicted element index and operation
- Ground truth element index is the position of the positive candidate after shuffling

### Step 3: Run and report

Run:
```bash
python scripts/ablation_single_agent.py --gpu 0 --output_dir outputs/ablation_single_agent
```

Expected output format (save as `outputs/ablation_single_agent/results.json`):
```json
{
    "experiment": "single_agent_browser_baseline",
    "backbone": "phi3_mini",
    "quantization": "4bit",
    "lora_r": 16,
    "num_train_tasks": "...",
    "num_test_tasks": 200,
    "epochs": 3,
    "results": {
        "elem_acc": 0.XXX,
        "op_f1": 0.XXX,
        "step_sr": 0.XXX
    },
    "comparison": {
        "full_system_vq": 0.815,
        "full_system_continuous": 0.783,
        "full_system_text": 0.780,
        "single_agent_baseline": "THIS RESULT"
    }
}
```

## What NOT to Do

- Do NOT use the `AgentTeam` class or any multi-agent infrastructure
- Do NOT load or initialize any communication modules, router, bitrate scheduler, or audit decoder
- Do NOT do SFT pretraining (Stage 1) or DPO (Stage 2) — go straight to benchmark FT
- Do NOT change the Mind2Web dataset processing or candidate shuffling — it must be identical
- Do NOT change hyperparameters (lr, epochs, batch size, LoRA config) — must match the paper's Stage 3

## Success Criteria

The script runs end-to-end on a single GPU with >=24GB VRAM and produces a JSON file with ElemAcc, OpF1, and StepSR on 200 Mind2Web test samples. The number tells us whether this paper's latent communication story holds up.
