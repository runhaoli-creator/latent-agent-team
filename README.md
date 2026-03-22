# Latent Agent Team

**Budgeted Latent Communication for Lightweight Multi-Agent Teams**

> A framework for coordinating multiple specialized LLM agents via learned latent communication channels instead of verbose natural-language messages, achieving state-of-the-art performance on web-agent benchmarks with dramatically lower communication cost.

---

## Overview

Latent Agent Team replaces the standard text-based inter-agent communication in multi-agent LLM systems with **learned latent channels** — either continuous embeddings or vector-quantized (VQ) codes. An **adaptive bitrate scheduler** dynamically allocates bandwidth based on task difficulty, while a **sparse router** determines which agents need to communicate at each step. An **audit decoder** reconstructs human-readable text from latent messages for interpretability.

### Key Results

| Benchmark   | Best Config (Phi-3 / VQ) | Previous SOTA        |
|-------------|--------------------------|----------------------|
| Mind2Web    | **81.5% ElemAcc**        | SeeAct GPT-4V 53.0% |
| WebShop     | **72.4% SR**             | WebAgent 58.1%       |
| AgentBench  | **66.8% SR**             | AutoGen GPT-4 45.2%  |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        AgentTeam                                │
│                                                                 │
│  ┌──────────┐  ┌───────────┐  ┌─────────┐  ┌──────────┐       │
│  │ Planner  │  │ Retriever │  │ Browser │  │ Verifier │       │
│  │ (T=0.7)  │  │  (FAISS)  │  │ (T=0.3) │  │ (T=0.1)  │       │
│  └────┬─────┘  └─────┬─────┘  └────┬────┘  └────┬─────┘       │
│       │              │              │             │              │
│       └──────────┬───┴──────────┬───┘             │              │
│                  │              │                  │              │
│           ┌──────▼──────┐ ┌────▼────┐  ┌─────────▼──────┐      │
│           │ Latent Comm │ │ Sparse  │  │   Memory       │      │
│           │ (Cont / VQ) │ │ Router  │  │  (ring buffer) │      │
│           └──────┬──────┘ └────┬────┘  └────────────────┘      │
│                  │             │                                 │
│           ┌──────▼─────────────▼──────┐                         │
│           │  Adaptive Bitrate Sched.  │                         │
│           │   K ∈ {4,8,16,32,64}      │                         │
│           └──────┬────────────────────┘                         │
│                  │                                              │
│           ┌──────▼──────┐                                       │
│           │ Audit Decoder│  (latent → text for interpretability)│
│           └─────────────┘                                       │
└─────────────────────────────────────────────────────────────────┘
```

### Agents

| Agent        | Role                               | Temperature | Max Tokens |
|--------------|-------------------------------------|-------------|------------|
| **Planner**  | Task decomposition & sub-goal gen   | 0.7         | 512        |
| **Retriever**| FAISS-backed dense exemplar retrieval | —          | —          |
| **Browser**  | Web/tool interaction execution      | 0.3         | 256        |
| **Verifier** | Constraint checking & validation    | 0.1         | 256        |
| **Memory**   | Episodic memory & compression       | 0.5         | 256        |

### Communication Modes

- **Continuous** — Dense latent embeddings passed between agents via learned linear projections
- **VQ (Vector-Quantized)** — 512-entry codebook with EMA updates, Gumbel-softmax relaxation, and straight-through estimator
- **Text** — Standard natural-language messages (baseline fallback)

### Core Model Components

| Module                     | Description                                              |
|----------------------------|----------------------------------------------------------|
| `BackboneManager`          | Loads 4-bit NF4 QLoRA backbones (Phi-3, Llama 3.2, Gemma 2, Ministral) |
| `RoleAdapter`              | Per-agent LoRA adapters (r=16, α=32)                     |
| `HiddenStateSummarizer`    | Pools last 16 positions → Linear(H→256) → query-based Transformer decoder |
| `LatentCommunicationModule`| Continuous or VQ channel for inter-agent messages         |
| `AdaptiveBitrateScheduler` | MLP(6→64→64→5) predicting bandwidth K ∈ {4,8,16,32,64}  |
| `SparseRouter`             | Per-pair MLP scoring with threshold-based gating          |
| `AuditDecoder`             | 2-layer seq2seq Transformer decoder for latent→text      |

## Project Structure

```
latent_agent_team/
├── configs/                          # Backbone configuration files
│   ├── phi3.yaml                     #   Phi-3 Mini 3.8B
│   ├── llama32_3b.yaml               #   Llama 3.2 3B
│   ├── gemma2_9b.yaml                #   Gemma 2 9B
│   └── ministral3_8b.yaml            #   Ministral 8B
├── src/latent_agent_team/            # Core library
│   ├── __init__.py
│   ├── team.py                       # AgentTeam orchestrator
│   ├── agents/                       # Specialized agent implementations
│   │   ├── planner.py
│   │   ├── retriever.py
│   │   ├── browser.py
│   │   ├── verifier.py
│   │   └── memory.py
│   ├── models/                       # Neural network modules
│   │   ├── backbone.py               #   QLoRA backbone loading
│   │   ├── role_adapter.py           #   Role-specific LoRA + summarizer
│   │   ├── latent_comm.py            #   Continuous & VQ channels
│   │   ├── router.py                 #   Bitrate scheduler + sparse router
│   │   └── audit_decoder.py          #   Latent → text audit trail
│   ├── train/                        # Training pipeline
│   │   ├── sft_bootstrap.py          #   Stage 1: SFT bootstrapping
│   │   ├── dpo_rollout.py            #   Stage 2: DPO preference opt.
│   │   └── eval.py                   #   Evaluation harness
│   ├── benchmarks/                   # Benchmark wrappers
│   │   ├── mind2web_wrapper.py
│   │   ├── webshop_wrapper.py
│   │   └── agentbench_wrapper.py
│   └── utils/                        # Shared utilities
│       ├── logger.py                 #   JSONL rollout recorder
│       └── metrics.py                #   Metric computation
├── scripts/                          # Training & evaluation scripts
│   ├── run_train.sh                  #   Full training pipeline
│   ├── run_eval.sh                   #   Evaluation on all benchmarks
│   ├── run_sft_training.py           #   SFT training launcher
│   ├── run_full_sft_pipeline.py      #   End-to-end SFT pipeline
│   ├── finetune_benchmarks.py        #   Benchmark-specific fine-tuning
│   ├── eval_finetuned.py             #   Evaluate fine-tuned checkpoints
│   ├── generate_teacher_traces.py    #   Generate teacher traces for SFT
│   └── download_data.sh              #   Dataset download utility
├── paper/                            # NeurIPS 2025 submission
│   ├── latent_agent_team.tex
│   └── neurips_2025.sty
├── pyproject.toml                    # Package build configuration
├── environment.yml                   # Conda environment specification
└── README.md
```

## Installation

### Prerequisites
- Python ≥ 3.11
- CUDA 12.1+
- 1–8 NVIDIA GPUs with ≥ 24 GB VRAM each (RTX 3090 / A100 / RTX 6000 recommended)

### Setup

```bash
# Clone the repository
git clone git@github.com:zhengtaoyao/latent-agent-team.git
cd latent-agent-team

# Option 1: Conda (recommended)
conda env create -f environment.yml
conda activate latent_agent_team

# Option 2: pip
pip install -e .

# Download benchmark datasets
bash scripts/download_data.sh
```

## Usage

### Training

The training pipeline consists of two stages:

**Stage 1: SFT Bootstrapping** — Teacher-forced training with a 6-term composite loss  
**Stage 2: DPO Preference Optimization** — Refine communication via preference pairs

```bash
# Full two-stage pipeline (default: Phi-3 backbone)
bash scripts/run_train.sh configs/phi3.yaml outputs/phi3_run

# With a different backbone
bash scripts/run_train.sh configs/gemma2_9b.yaml outputs/gemma_run

# Stage 1 only (SFT)
python scripts/run_sft_training.py --config configs/phi3.yaml --output_dir outputs/sft_run

# End-to-end pipeline (trace generation → SFT → eval)
python scripts/run_full_sft_pipeline.py --config configs/phi3.yaml --output_dir outputs/full_run
```

### Evaluation

```bash
# Evaluate a trained checkpoint on all benchmarks
bash scripts/run_eval.sh configs/phi3.yaml outputs/phi3_run/stage2 results/phi3

# Evaluate on specific benchmarks
BENCHMARKS=mind2web,webshop bash scripts/run_eval.sh configs/phi3.yaml outputs/phi3_run/stage2 results/phi3

# Evaluate fine-tuned model
python scripts/eval_finetuned.py --config configs/phi3.yaml --checkpoint outputs/phi3_run/stage2
```

### Configuration

Each backbone config (in `configs/`) controls:

| Section              | Key Parameters                                          |
|----------------------|---------------------------------------------------------|
| `backbone`           | HuggingFace model ID, hidden size, quantization, dtype  |
| `lora`               | Rank (16), alpha (32), dropout (0.05), target modules   |
| `communication`      | Mode (continuous/vq/text), latent dim (256), codebook   |
| `bitrate_scheduler`  | K choices [4,8,16,32,64], entropy thresholds            |
| `sparse_router`      | Routing threshold (0.5), hidden dim (128)               |
| `audit_decoder`      | Decoder dim (256), max decode length (32)               |
| `sft`                | Epochs (3), LR (2e-4), loss weights                     |
| `dpo`                | Epochs (1), LR (5e-5), β (0.1)                         |
| `eval`               | Benchmarks, max tasks per benchmark                     |

To switch communication mode, edit `communication.mode` in the config:

```yaml
communication:
  mode: "vq"  # Options: continuous, vq, text
```

### Programmatic Usage

```python
from latent_agent_team import AgentTeam, EpisodeResult

# Initialize with a config file
team = AgentTeam(config_path="configs/phi3.yaml")

# Run a single episode
result: EpisodeResult = team.step(
    task="Find the cheapest round-trip flight from NYC to London in December",
    environment=env,
)

print(f"Success: {result.success}")
print(f"Communication cost: {result.comm_cost:.2f} tokens equivalent")
print(f"Audit trail: {result.audit_text}")
```

## Supported Backbones

| Backbone         | Parameters | Hidden Size | Config File            |
|------------------|------------|-------------|------------------------|
| Phi-3 Mini       | 3.8B       | 3072        | `configs/phi3.yaml`    |
| Llama 3.2        | 3B         | 3072        | `configs/llama32_3b.yaml` |
| Gemma 2          | 9B         | 3584        | `configs/gemma2_9b.yaml` |
| Ministral        | 8B         | 4096        | `configs/ministral3_8b.yaml` |

All backbones are loaded with **4-bit NF4 quantization** via `bitsandbytes` and fine-tuned with **QLoRA** adapters.

## Training Details

### Stage 1: SFT Bootstrapping

6-term composite loss:

$$\mathcal{L}_{\text{SFT}} = \mathcal{L}_{\text{action}} + 0.3 \cdot \mathcal{L}_{\text{constraint}} + 0.1 \cdot \mathcal{L}_{\text{audit}} + 0.2 \cdot \mathcal{L}_{\text{route}} + 0.01 \cdot \mathcal{L}_{\text{bitrate}} + \mathcal{L}_{\text{VQ}}$$

- AdamW optimizer, LR = 2×10⁻⁴, 3 epochs

### Stage 2: DPO Preference Optimization

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)} - \beta \log \frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)}\right)\right]$$

- β = 0.1, LR = 5×10⁻⁵, 1 epoch

## Citation

```bibtex
@inproceedings{latent_agent_team_2025,
  title={Budgeted Latent Communication for Lightweight Multi-Agent Teams},
  author={Latent Agent Team Authors},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025}
}
```

## License

MIT License — see [pyproject.toml](pyproject.toml) for details.
