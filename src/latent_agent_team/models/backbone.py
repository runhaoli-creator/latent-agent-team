"""
backbone.py — Shared frozen backbone loading with QLoRA / 4-bit quantization.

Supports: Phi-3 Mini 3.8B, Llama 3.2 3B Instruct, Gemma 2 9B, Ministral 3 8B.
All base weights are frozen; only LoRA adapters and communication modules are trained.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)

logger = logging.getLogger(__name__)

# ── Registry of supported backbones ──────────────────────────────────────────

BACKBONE_REGISTRY: Dict[str, Dict[str, Any]] = {
    "phi3_mini": {
        "hf_id": "microsoft/Phi-3-mini-4k-instruct",
        "hidden_size": 3072,
        "num_layers": 32,
        "description": "Phi-3 Mini 3.8B — cheapest primary backbone",
        "lora_target_modules": ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
    },
    "llama32_3b": {
        "hf_id": "meta-llama/Llama-3.2-3B-Instruct",
        "hidden_size": 3072,
        "num_layers": 28,
        "description": "Llama 3.2 3B Instruct — strong small agentic baseline",
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    },
    "gemma2_9b": {
        "hf_id": "google/gemma-2-9b-it",
        "hidden_size": 3584,
        "num_layers": 42,
        "description": "Gemma 2 9B — higher-capacity text model",
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    },
    "ministral_8b": {
        "hf_id": "mistralai/Ministral-8B-Instruct-2410",
        "hidden_size": 4096,
        "num_layers": 36,
        "description": "Ministral 3 8B — replaces retired Mistral 7B per Mistral docs",
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    },
}


@dataclass
class BackboneConfig:
    """Configuration for loading a backbone model."""
    backbone_name: str = "phi3_mini"
    quantization: str = "4bit"  # "4bit", "8bit", "none"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    max_seq_len: int = 4096
    dtype: str = "bfloat16"
    trust_remote_code: bool = False  # transformers 4.57+ has native support for Phi-3, Llama, Gemma, Mistral
    use_flash_attention: bool = True
    gradient_checkpointing: bool = True
    device_map: str = "auto"
    # Extra HF kwargs
    extra_model_kwargs: Dict[str, Any] = field(default_factory=dict)


def _get_bnb_config(quantization: str) -> Optional[BitsAndBytesConfig]:
    """Return BitsAndBytesConfig for 4-bit or 8-bit quantization."""
    if quantization == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif quantization == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def load_backbone(
    cfg: BackboneConfig,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase, Dict[str, Any]]:
    """
    Load a frozen backbone with optional quantization and prepare for LoRA.

    Returns:
        (model, tokenizer, backbone_info)
    """
    if cfg.backbone_name not in BACKBONE_REGISTRY:
        raise ValueError(
            f"Unknown backbone '{cfg.backbone_name}'. "
            f"Choose from: {list(BACKBONE_REGISTRY.keys())}"
        )

    info = BACKBONE_REGISTRY[cfg.backbone_name]
    hf_id = info["hf_id"]
    logger.info(f"Loading backbone: {info['description']}  ({hf_id})")

    # ── Tokenizer ────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        hf_id,
        trust_remote_code=cfg.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── Quantization config ──────────────────────────────────────────────
    bnb_config = _get_bnb_config(cfg.quantization)

    # ── Torch dtype ──────────────────────────────────────────────────────
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    compute_dtype = dtype_map.get(cfg.dtype, torch.bfloat16)

    # ── Load model ───────────────────────────────────────────────────────
    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": cfg.trust_remote_code,
        "torch_dtype": compute_dtype,
        "device_map": cfg.device_map,
    }
    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config
    if cfg.use_flash_attention:
        try:
            import flash_attn  # noqa: F401
            model_kwargs["attn_implementation"] = "flash_attention_2"
        except ImportError:
            logger.warning("flash_attn not installed, will try sdpa or eager attention")
    model_kwargs.update(cfg.extra_model_kwargs)

    try:
        model = AutoModelForCausalLM.from_pretrained(hf_id, **model_kwargs)
    except (ValueError, ImportError) as e:
        # Some models (e.g., Phi-3 custom code) don't support sdpa/flash_attention_2;
        # fall back to eager attention.
        logger.warning("Model load failed with attn impl '%s': %s — retrying with eager",
                       model_kwargs.get("attn_implementation", "default"), e)
        model_kwargs["attn_implementation"] = "eager"
        model = AutoModelForCausalLM.from_pretrained(hf_id, **model_kwargs)

    # ── Freeze all base weights ──────────────────────────────────────────
    for param in model.parameters():
        param.requires_grad = False

    # ── Prepare for k-bit training (handles gradient issues with quant) ──
    if cfg.quantization in ("4bit", "8bit"):
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=cfg.gradient_checkpointing,
        )
    elif cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    logger.info(
        f"Backbone loaded: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params, "
        f"all frozen. Quantization={cfg.quantization}"
    )

    backbone_info = {
        **info,
        "hidden_size": info["hidden_size"],
        "num_layers": info["num_layers"],
        "quantization": cfg.quantization,
    }
    return model, tokenizer, backbone_info


class BackboneManager:
    """
    Manages the shared backbone + multiple role-specific LoRA adapters.

    The backbone is loaded once; adapters are swapped per agent role.
    """

    def __init__(self, cfg: BackboneConfig):
        self.cfg = cfg
        self.model, self.tokenizer, self.info = load_backbone(cfg)
        self._adapters: Dict[str, LoraConfig] = {}
        self._active_adapter: Optional[str] = None

    @property
    def hidden_size(self) -> int:
        return self.info["hidden_size"]

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def add_adapter(self, role_name: str, lora_config: Optional[LoraConfig] = None) -> None:
        """Register a LoRA adapter for a specific agent role."""
        if lora_config is None:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.cfg.lora_r,
                lora_alpha=self.cfg.lora_alpha,
                lora_dropout=self.cfg.lora_dropout,
                target_modules=self.info["lora_target_modules"],
                bias="none",
            )
        if not hasattr(self.model, "peft_config"):
            # First adapter — wrap with PEFT
            self.model = get_peft_model(self.model, lora_config, adapter_name=role_name)
        else:
            self.model.add_adapter(role_name, lora_config)
        self._adapters[role_name] = lora_config
        logger.info(f"Added LoRA adapter '{role_name}' (r={lora_config.r})")

    def set_active_adapter(self, role_name: str) -> None:
        """Activate the adapter for the given role."""
        if role_name not in self._adapters:
            raise ValueError(f"No adapter registered for role '{role_name}'")
        self.model.set_adapter(role_name)
        self._active_adapter = role_name

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None,
        output_hidden_states: bool = True,
        **kwargs,
    ):
        """Run forward pass through backbone with the active adapter."""
        # Ensure inputs are on the model's device
        model_device = self.device
        if input_ids.device != model_device:
            input_ids = input_ids.to(model_device)
        if attention_mask is not None and attention_mask.device != model_device:
            attention_mask = attention_mask.to(model_device)
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )

    def get_hidden_states(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extract final-layer hidden states from the backbone."""
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        # outputs.hidden_states is a tuple of (num_layers + 1) tensors
        return outputs.hidden_states[-1]  # [B, seq_len, hidden_size]

    def trainable_parameters(self):
        """Yield only trainable parameters (LoRA adapters)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                yield name, param

    def count_trainable(self) -> int:
        return sum(p.numel() for _, p in self.trainable_parameters())

    def save_adapters(self, save_dir: str) -> None:
        """Save all LoRA adapters."""
        for name in self._adapters:
            self.model.set_adapter(name)
            self.model.save_pretrained(f"{save_dir}/{name}")
        logger.info(f"Saved {len(self._adapters)} adapters to {save_dir}")

    def load_adapters(self, load_dir: str) -> None:
        """Load LoRA adapters from disk."""
        import os
        for name in os.listdir(load_dir):
            adapter_path = os.path.join(load_dir, name)
            if os.path.isdir(adapter_path):
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(
                    self.model, adapter_path, adapter_name=name
                )
                self._adapters[name] = None
                logger.info(f"Loaded adapter '{name}' from {adapter_path}")
