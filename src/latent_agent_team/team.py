"""
team.py — AgentTeam orchestrator.

Ties together all 5 agents, the latent communication module,
adaptive bitrate scheduler, sparse router, and audit decoder
into a single runnable unit.

Usage:
    team = AgentTeam.from_config(cfg)
    result = team.run_episode(instruction, env, max_steps=15)
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from .agents.planner import PlannerAgent
from .agents.retriever import RetrieverAgent
from .agents.browser import BrowserAgent
from .agents.verifier import VerifierAgent
from .agents.memory import MemoryManager
from .models.backbone import BackboneManager, BackboneConfig, BACKBONE_REGISTRY
from .models.role_adapter import RoleAdapterManager
from .models.latent_comm import LatentCommunicationModule, CommModuleConfig
from .models.router import AdaptiveBitrateScheduler, SparseRouter, AGENT_NAMES
from .models.audit_decoder import AuditDecoder, AuditRecord

logger = logging.getLogger(__name__)


# ── Episode Result ─────────────────────────────────────────────────────────────

@dataclass
class EpisodeResult:
    task_instruction: str
    success: bool
    reward: float
    total_steps: int
    total_k_used: int
    mean_k: float
    num_routing_edges: int
    actions: List[Dict]
    audit_records: List[Dict]
    failure_tags: List[str]
    elapsed_sec: float


# ── Agent Team ─────────────────────────────────────────────────────────────────

class AgentTeam:
    """
    Full 5-agent team with latent communication.
    """

    COMM_EDGES: List[Tuple[str, str]] = [
        ("planner",   "browser"),
        ("planner",   "retriever"),
        ("retriever", "browser"),
        ("browser",   "verifier"),
        ("verifier",  "planner"),
        ("memory",    "planner"),
    ]

    ROLE_TO_IDX = {name: i for i, name in enumerate(AGENT_NAMES)}

    def __init__(
        self,
        backbone: BackboneManager,
        role_manager: RoleAdapterManager,
        planner: PlannerAgent,
        retriever: RetrieverAgent,
        browser: BrowserAgent,
        verifier: VerifierAgent,
        memory: MemoryManager,
        comm_module: LatentCommunicationModule,
        bitrate_scheduler: AdaptiveBitrateScheduler,
        router: SparseRouter,
        audit_decoder: AuditDecoder,
        device: Optional[torch.device] = None,
    ):
        self.backbone = backbone
        self.role_manager = role_manager
        self.planner = planner
        self.retriever = retriever
        self.browser = browser
        self.verifier = verifier
        self.memory = memory
        self.comm = comm_module
        self.bitrate = bitrate_scheduler
        self.router = router
        self.audit = audit_decoder
        self.device = device or torch.device("cpu")
        self._step_idx: int = 0
        self._task_instruction: str = ""

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, cfg: Any, device: Optional[torch.device] = None) -> "AgentTeam":
        """Build a complete AgentTeam from an OmegaConf / dict config."""
        from omegaconf import OmegaConf
        if not isinstance(cfg, dict):
            cfg = OmegaConf.to_container(cfg, resolve=True)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── Backbone ───────────────────────────────────────────────────
        bb_cfg = cfg["backbone"]
        backbone_name = bb_cfg.get("name", bb_cfg.get("model_name", "phi3_mini"))
        if backbone_name not in BACKBONE_REGISTRY:
            raise ValueError(f"Unknown backbone '{backbone_name}'. Available: {list(BACKBONE_REGISTRY.keys())}")

        backbone_cfg = BackboneConfig(
            backbone_name=backbone_name,
            quantization=bb_cfg.get("quantization", "4bit"),
            max_seq_len=bb_cfg.get("max_seq_len", 4096),
            dtype=bb_cfg.get("dtype", "bfloat16"),
            use_flash_attention=bb_cfg.get("use_flash_attention", True),
            gradient_checkpointing=bb_cfg.get("gradient_checkpointing", True),
            device_map=bb_cfg.get("device_map", "auto"),
        )
        # Apply LoRA settings from config
        lora_cfg = cfg.get("lora", {})
        backbone_cfg.lora_r = lora_cfg.get("r", 16)
        backbone_cfg.lora_alpha = lora_cfg.get("alpha", 32)
        backbone_cfg.lora_dropout = lora_cfg.get("dropout", 0.05)

        backbone = BackboneManager(backbone_cfg)

        # Register LoRA adapters for each role
        roles = ["planner", "retriever", "browser", "verifier", "memory"]
        for role in roles:
            backbone.add_adapter(role)

        # ── Role Adapters ─────────────────────────────────────────────
        hidden_size = backbone.hidden_size
        role_manager = RoleAdapterManager(hidden_size=hidden_size, roles=roles)
        role_manager.to(device)

        # ── Comm Module ────────────────────────────────────────────────
        comm_cfg_dict = cfg.get("communication", {})
        comm_config = CommModuleConfig(
            mode=comm_cfg_dict.get("mode", "continuous"),
            hidden_size=hidden_size,
            latent_dim=comm_cfg_dict.get("latent_dim", 256),
            codebook_size=comm_cfg_dict.get("codebook_size", 512),
            commitment_cost=comm_cfg_dict.get("commitment_cost", 0.25),
            ema_decay=comm_cfg_dict.get("ema_decay", 0.99),
            use_gumbel_warmup=comm_cfg_dict.get("use_gumbel_warmup", True),
            injection_mode=comm_cfg_dict.get("injection_mode", "prefix"),
        )
        comm_module = LatentCommunicationModule(comm_config).to(device)

        # ── Bitrate Scheduler ─────────────────────────────────────────
        bs_cfg = cfg.get("bitrate_scheduler", {})
        bitrate_scheduler = AdaptiveBitrateScheduler(
            hidden_dim=bs_cfg.get("hidden_dim", 64),
        ).to(device)

        # ── Sparse Router ─────────────────────────────────────────────
        sr_cfg = cfg.get("sparse_router", {})
        latent_dim = comm_cfg_dict.get("latent_dim", 256)
        router = SparseRouter(
            hidden_size=hidden_size,
            router_hidden_dim=sr_cfg.get("router_hidden_dim", 128),
            routing_threshold=sr_cfg.get("routing_threshold", 0.5),
            min_recipients=sr_cfg.get("min_recipients", 1),
        ).to(device)

        # ── Audit Decoder ─────────────────────────────────────────
        ad_cfg = cfg.get("audit_decoder", {})
        # Use model config vocab_size (includes added/special tokens + padding)
        effective_vocab_size = backbone.model.config.vocab_size
        audit_decoder = AuditDecoder(
            hidden_size=hidden_size,
            vocab_size=effective_vocab_size,
            decoder_dim=ad_cfg.get("decoder_dim", 256),
            max_decode_len=ad_cfg.get("max_decode_len", 32),
        ).to(device)

        # ── Agents ────────────────────────────────────────────────────
        planner = PlannerAgent(
            backbone=backbone,
            role_manager=role_manager,
            comm_module=comm_module,
            bitrate_scheduler=bitrate_scheduler,
            sparse_router=router,
        )
        retriever = RetrieverAgent(
            backbone=backbone,
            role_manager=role_manager,
            comm_module=comm_module,
        )
        browser = BrowserAgent(
            backbone=backbone,
            role_manager=role_manager,
            comm_module=comm_module,
        )
        verifier = VerifierAgent(
            backbone=backbone,
            role_manager=role_manager,
            comm_module=comm_module,
        )
        memory_agent = MemoryManager(
            backbone=backbone,
            role_manager=role_manager,
            comm_module=comm_module,
        )

        return cls(
            backbone=backbone,
            role_manager=role_manager,
            planner=planner,
            retriever=retriever,
            browser=browser,
            verifier=verifier,
            memory=memory_agent,
            comm_module=comm_module,
            bitrate_scheduler=bitrate_scheduler,
            router=router,
            audit_decoder=audit_decoder,
            device=device,
        )

    # ── Reset ────────────────────────────────────────────────────────────────

    def reset(self, task_instruction: str = "") -> None:
        self._step_idx = 0
        self._task_instruction = task_instruction
        self.planner.reset(task_instruction=task_instruction)
        self.browser.reset()
        self.verifier.reset()
        ep_id = f"ep_{uuid.uuid4().hex[:8]}"
        self.memory.reset(episode_id=ep_id)

    # ── Single Team Step ──────────────────────────────────────────────────────

    def step(
        self,
        observation: str,
        task_instruction: str,
        k_budget: Optional[int] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        t0 = time.time()
        step_info: Dict[str, Any] = {
            "step_idx": self._step_idx,
            "routes": [],
            "k_values": {},
            "audit_texts": {},
            "codebook_ids": {},
        }

        # ── 1. Memory recall ──────────────────────────────────────────
        recalled = self.memory.recall(observation, n=3)
        memory_context = "\n".join(
            e.observation[:100] for e in recalled[:2]
        )

        # ── 2. Planner step → (action, hidden_states, entropy) ────────
        plan_action, planner_hidden, planner_entropy = self.planner.step(
            observation=observation,
            incoming_messages=[{"sender": "memory", "content": memory_context}] if memory_context else None,
            device=self.device,
        )
        sub_goal = ""
        if isinstance(plan_action, dict):
            sgs = plan_action.get("sub_goals", [])
            sub_goal = sgs[0] if sgs else plan_action.get("reasoning", task_instruction)
        # Ensure sub_goal is a string (planner may return non-string)
        if not isinstance(sub_goal, str):
            sub_goal = str(sub_goal) if sub_goal else task_instruction

        # ── 3. Retriever step → (action, hidden_states, confidence) ───
        retriever_action, retriever_hidden, retrieval_confidence = self.retriever.step(
            query=sub_goal or task_instruction,
            device=self.device,
        )
        top_evidence = retriever_action.get("top_result", "")

        # ── 4. Bitrate scheduling ─────────────────────────────────────
        if k_budget is not None:
            k = k_budget
        else:
            k = self.bitrate.select_k(
            planner_entropy=planner_entropy,
            verifier_disagreement=self.verifier.last_disagreement,
            retrieval_confidence=retrieval_confidence,
            recent_failure_count=self.memory.memory_store.failure_pattern_analysis().get("total", 0),
            task_progress=min(self._step_idx / 15.0, 1.0),
            step_budget_remaining=1.0,
            device=self.device,
        )
        step_info["k_values"]["selected_k"] = k

        # ── 5. Latent communication: summarize + route ────────────────
        # Summarize planner hidden states into K latent tokens
        planner_latent = self.planner.generate_latent_message(
            planner_hidden, k=min(k, 64)
        )  # [B, K, hidden_size]

        # Route: decide which agents receive planner's message
        sender_idx = self.ROLE_TO_IDX.get("planner", 0)
        routing_probs, routing_mask = self.router(
            sender_idx=sender_idx,
            latent_tokens=planner_latent,
            hard=True,
        )
        recipient_names = self.router.get_recipient_names(routing_mask)
        step_info["routes"] = [f"planner→{r}" for r in recipient_names]

        # Pass through comm module for VQ / continuous encoding
        # Use real receiver observation embeddings for proper latent injection
        obs_tokens = self.backbone.tokenizer(
            observation[:400], return_tensors="pt", truncation=True, max_length=128,
        )
        obs_input_ids = obs_tokens["input_ids"].to(self.device)
        with torch.no_grad():
            receiver_embeds = self.backbone.model.get_input_embeddings()(obs_input_ids)
        combined, mask, comm_info = self.comm(
            hidden_summary=planner_latent,
            obs_embeds=receiver_embeds,
            training=self.comm.training,
        )
        step_info["codebook_ids"]["planner"] = comm_info.get("indices", None)

        # Audit decode
        audit_texts = []
        try:
            audit_texts = self.audit.generate_audit_text(
                planner_latent, self.backbone.tokenizer, max_len=32
            )
            step_info["audit_texts"]["planner"] = audit_texts[0] if audit_texts else ""
        except Exception as e:
            logger.debug("Audit decode error: %s", e)

        # ── 6. Browser step ───────────────────────────────────────────
        incoming_latent = planner_latent if "browser" in recipient_names else None
        browser_action, browser_hidden = self.browser.step(
            observation=observation,
            sub_goal=sub_goal,
            retrieved_evidence=top_evidence,
            incoming_latent=incoming_latent,
            device=self.device,
        )

        # ── 7. Verifier step ──────────────────────────────────────────
        verif_result, verifier_hidden = self.verifier.step(
            observation=observation,
            task_goal=task_instruction,
            proposed_action=json.dumps(browser_action) if isinstance(browser_action, dict) else str(browser_action),
            device=self.device,
        )
        step_info["verification"] = verif_result.status
        step_info["disagreement"] = verif_result.disagreement_score

        # ── 8. Memory store ───────────────────────────────────────────
        is_failure = verif_result.status == "fail"
        self.memory.store(
            step=self._step_idx,
            observation=observation[:200],
            action=browser_action,
            reward=1.0 if verif_result.status == "pass" else 0.0,
            sender_role="browser",
            is_failure=is_failure,
        )

        # Finalize
        total_k = k
        step_info["total_k"] = total_k
        step_info["elapsed_ms"] = (time.time() - t0) * 1000
        step_info["action"] = browser_action

        self._step_idx += 1
        return browser_action, step_info

    def enable_fast_eval(self):
        """Optimize for fast evaluation: reduce token budgets, skip heavy ops."""
        # Reduce max_new_tokens for each agent
        self.planner.max_new_tokens = 128   # was 512
        self.browser.max_new_tokens = 64    # was 256
        self.verifier.max_new_tokens = 48   # was 256
        if hasattr(self.memory, 'max_new_tokens'):
            self.memory.max_new_tokens = 64
        self._fast_eval = True
        logger.info("Fast eval enabled: planner=128, browser=64, verifier=48 max_new_tokens")

    def step_fast(
        self,
        observation: str,
        task_instruction: str,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Fast step for evaluation: skips verifier and audit decode."""
        t0 = time.time()
        step_info: Dict[str, Any] = {
            "step_idx": self._step_idx,
            "routes": [],
            "k_values": {},
        }

        # 1. Memory recall (no model call)
        recalled = self.memory.recall(observation, n=3)
        memory_context = "\n".join(e.observation[:100] for e in recalled[:2])

        # 2. Planner step
        plan_action, planner_hidden, planner_entropy = self.planner.step(
            observation=observation,
            incoming_messages=[{"sender": "memory", "content": memory_context}] if memory_context else None,
            device=self.device,
        )
        sub_goal = ""
        if isinstance(plan_action, dict):
            sgs = plan_action.get("sub_goals", [])
            sub_goal = sgs[0] if sgs else plan_action.get("reasoning", task_instruction)
        if not isinstance(sub_goal, str):
            sub_goal = str(sub_goal) if sub_goal else task_instruction

        # 3. Retriever step (forward pass only, no generate)
        retriever_action, retriever_hidden, retrieval_confidence = self.retriever.step(
            query=sub_goal or task_instruction,
            device=self.device,
        )
        top_evidence = retriever_action.get("top_result", "")

        # 4. Bitrate scheduling
        k = self.bitrate.select_k(
            planner_entropy=planner_entropy,
            verifier_disagreement=0.0,  # skip verifier feedback
            retrieval_confidence=retrieval_confidence,
            recent_failure_count=0,
            task_progress=min(self._step_idx / 15.0, 1.0),
            step_budget_remaining=1.0,
            device=self.device,
        )
        step_info["k_values"]["selected_k"] = k

        # 5. Latent communication (planner → browser)
        planner_latent = self.planner.generate_latent_message(
            planner_hidden, k=min(k, 64)
        )
        sender_idx = self.ROLE_TO_IDX.get("planner", 0)
        routing_probs, routing_mask = self.router(
            sender_idx=sender_idx,
            latent_tokens=planner_latent,
            hard=True,
        )
        recipient_names = self.router.get_recipient_names(routing_mask)
        step_info["routes"] = [f"planner→{r}" for r in recipient_names]

        # Comm encode with real receiver embeddings
        obs_tokens = self.backbone.tokenizer(
            observation[:400], return_tensors="pt", truncation=True, max_length=128,
        )
        obs_input_ids = obs_tokens["input_ids"].to(self.device)
        with torch.no_grad():
            receiver_embeds = self.backbone.model.get_input_embeddings()(obs_input_ids)
        combined, mask, comm_info = self.comm(
            hidden_summary=planner_latent,
            obs_embeds=receiver_embeds,
            training=False,
        )

        # 6. Browser step (main generation)
        incoming_latent = planner_latent if "browser" in recipient_names else None
        browser_action, browser_hidden = self.browser.step(
            observation=observation,
            sub_goal=sub_goal,
            retrieved_evidence=top_evidence,
            incoming_latent=incoming_latent,
            device=self.device,
        )

        # SKIP verifier (saves ~30% time)
        # SKIP audit decode (saves ~5% time)

        # 7. Memory store (no model call)
        self.memory.store(
            step=self._step_idx,
            observation=observation[:200],
            action=browser_action,
            reward=0.5,  # neutral since no verifier
            sender_role="browser",
            is_failure=False,
        )

        step_info["total_k"] = k
        step_info["elapsed_ms"] = (time.time() - t0) * 1000
        step_info["action"] = browser_action
        step_info["verification"] = "skipped"
        step_info["disagreement"] = 0.0

        self._step_idx += 1
        return browser_action, step_info

    # ── Full Episode ──────────────────────────────────────────────────────────

    def run_episode(
        self,
        instruction: str,
        env: Any,
        max_steps: int = 15,
    ) -> EpisodeResult:
        t_start = time.time()
        self.reset(task_instruction=instruction)

        obs_dict = env.reset()
        if isinstance(obs_dict, dict):
            obs_text = obs_dict.get("observation", str(obs_dict))
        else:
            obs_text = str(obs_dict)

        actions: List[Dict] = []
        all_audit: List[Dict] = []
        total_k = 0
        num_routes = 0
        done = False
        reward = 0.0
        failure_tags: List[str] = []

        for step_num in range(max_steps):
            step_fn = self.step_fast if getattr(self, '_fast_eval', False) else self.step
            action, step_info = step_fn(obs_text, instruction)
            actions.append(action)
            total_k += step_info.get("total_k", 0)
            num_routes += len(step_info.get("routes", []))
            all_audit.append({
                "step": step_num,
                "routes": step_info.get("routes", []),
                "k": step_info.get("total_k", 0),
                "audit_texts": step_info.get("audit_texts", {}),
                "verification": step_info.get("verification", "uncertain"),
            })

            # Env step
            try:
                env_result = env.step(action)
                if isinstance(env_result, tuple):
                    if len(env_result) == 4:
                        obs_dict, r, done, info = env_result
                    elif len(env_result) == 3:
                        obs_dict, r, done = env_result
                        info = {}
                    else:
                        obs_dict, r = env_result[:2]
                        done = False
                        info = {}
                    reward += float(r)
                elif isinstance(env_result, dict):
                    obs_dict = env_result
                    reward += float(env_result.get("reward", 0.0))
                    done = env_result.get("done", False)
                else:
                    obs_dict = {"observation": str(env_result)}
                    done = False

                if isinstance(obs_dict, dict):
                    obs_text = obs_dict.get("observation", obs_dict.get("obs", str(obs_dict)))
                    if "failure_tag" in obs_dict:
                        failure_tags.append(obs_dict["failure_tag"])
                else:
                    obs_text = str(obs_dict)

            except Exception as e:
                logger.error("Env step error at step %d: %s", step_num, e)
                failure_tags.append(f"env_error:{type(e).__name__}")
                done = True
                break

            if done:
                break

        success = reward >= 0.5  # Partial success threshold (50%+ attribute match)
        mean_k = total_k / max(1, len(actions))

        return EpisodeResult(
            task_instruction=instruction,
            success=success,
            reward=reward,
            total_steps=len(actions),
            total_k_used=total_k,
            mean_k=mean_k,
            num_routing_edges=num_routes,
            actions=actions,
            audit_records=all_audit,
            failure_tags=failure_tags,
            elapsed_sec=time.time() - t_start,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_failure_rate(self) -> float:
        try:
            patterns = self.memory.memory_store.failure_pattern_analysis()
            return sum(patterns.values()) / max(1, len(self.memory.memory_store.entries))
        except Exception:
            return 0.0

    # ── Utility ───────────────────────────────────────────────────────────────

    def trainable_parameters(self) -> List[torch.nn.Parameter]:
        params = []
        for mod in [self.comm, self.bitrate, self.router, self.audit]:
            params.extend(p for p in mod.parameters() if p.requires_grad)
        # Role adapter summarizers
        for role_name, adapter in self.role_manager.adapters.items():
            params.extend(p for p in adapter.summarizer.parameters() if p.requires_grad)
        return params

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def save_checkpoint(self, path: str) -> None:
        import os
        os.makedirs(path, exist_ok=True)
        torch.save(self.comm.state_dict(), os.path.join(path, "comm.pt"))
        torch.save(self.bitrate.state_dict(), os.path.join(path, "bitrate.pt"))
        torch.save(self.router.state_dict(), os.path.join(path, "router.pt"))
        torch.save(self.audit.state_dict(), os.path.join(path, "audit.pt"))
        self.backbone.save_adapters(path)
        logger.info("Checkpoint saved to %s", path)

    def load_checkpoint(self, path: str, strict: bool = False) -> None:
        import os
        def _load(name: str, module: torch.nn.Module) -> None:
            fp = os.path.join(path, f"{name}.pt")
            if os.path.exists(fp):
                module.load_state_dict(torch.load(fp, map_location=self.device), strict=strict)
                logger.info("Loaded %s from %s", name, fp)
        _load("comm", self.comm)
        _load("bitrate", self.bitrate)
        _load("router", self.router)
        _load("audit", self.audit)
        self.backbone.load_adapters(path)
        logger.info("Checkpoint loaded from %s", path)
