"""Pydantic schemas for the latent-agent-team serving API.

Exposes the 5-agent team (Planner, Retriever, Browser, Verifier, Memory)
over HTTP so benchmark runners and interactive debuggers can send an
instruction + environment spec and receive an episode trace, the latent
communication audit trail, and per-agent step counts.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

CommMode = Literal["continuous", "vq", "text"]
BenchmarkName = Literal["mind2web", "webshop", "agentbench"]


class RunEpisodeRequest(BaseModel):
    instruction: str = Field(..., min_length=1)
    env_spec: dict[str, Any] = Field(default_factory=dict,
        description="Serialized environment state / start page / task metadata.")
    max_steps: int = Field(15, ge=1, le=64)
    comm_mode: CommMode = "vq"
    bitrate_k: int | None = Field(None, description="Override scheduler; {4,8,16,32,64}.")
    seed: int | None = None
    emit_audit: bool = Field(True, description="Emit latent->text audit decoder trace.")


class AgentStep(BaseModel):
    step: int
    agent: Literal["planner", "retriever", "browser", "verifier", "memory"]
    action: str
    latent_bits: int | None = None
    audit: str | None = None


class RunEpisodeResponse(BaseModel):
    episode_id: str
    success: bool
    final_answer: str | None
    steps: list[AgentStep]
    total_latent_bits: int
    average_bitrate: float
    latency_ms: float
    comm_mode: CommMode


class BenchmarkRunRequest(BaseModel):
    benchmark: BenchmarkName
    split: Literal["train", "dev", "test"] = "test"
    max_examples: int = Field(50, ge=1, le=2000)
    concurrency: int = Field(4, ge=1, le=32)
    comm_mode: CommMode = "vq"


class BenchmarkRunResponse(BaseModel):
    benchmark: BenchmarkName
    n_examples: int
    metric_name: str
    metric_value: float
    success_rate: float
    avg_latent_bits_per_step: float
    wall_time_seconds: float


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    team_ready: bool
    backbone: str | None
    comm_modes_available: list[CommMode]
    cache_backend: str


class MetricsResponse(BaseModel):
    episodes_total: int
    success_total: int
    success_rate: float
    avg_bitrate: float
    p95_episode_latency_ms: float
