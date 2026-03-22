"""
latent_agent_team — Budgeted Latent Communication for Lightweight Multi-Agent Teams.

Package layout
--------------
latent_agent_team/
    models/
        backbone.py       — shared backbone loading + QLoRA
        role_adapter.py   — per-role LoRA adapters + hidden-state summariser
        latent_comm.py    — continuous & VQ latent channels
        router.py         — adaptive bitrate scheduler + sparse router
        audit_decoder.py  — latent → text audit trail
    agents/
        planner.py        — task decomposition
        retriever.py      — FAISS-backed dense retrieval
        browser.py        — web/tool interaction
        verifier.py       — constraint checking
        memory.py         — episodic memory + compression
    train/
        sft_bootstrap.py  — Stage 1: supervised communication bootstrapping
        dpo_rollout.py    — Stage 2: DPO preference optimisation
        eval.py           — comprehensive evaluation + baselines
    benchmarks/
        mind2web_wrapper.py
        webshop_wrapper.py
        agentbench_wrapper.py
    team.py               — AgentTeam orchestrator
    utils/
        logger.py         — structured JSONL rollout recorder
        metrics.py        — shared metric utilities
"""

__version__ = "0.1.0"
__author__  = "Latent Agent Team"

from .team import AgentTeam, EpisodeResult

__all__ = ["AgentTeam", "EpisodeResult"]
