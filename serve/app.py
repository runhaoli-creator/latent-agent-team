"""FastAPI service wrapping the 5-agent team orchestrator.

Routes:

* ``GET  /health``              — liveness, backbone, available comm modes
* ``GET  /metrics``              — episode / bitrate / latency counters
* ``POST /v1/run_episode``       — run a single instruction end-to-end
* ``POST /v1/run_benchmark``     — run a bounded slice of a benchmark with
                                   concurrent workers (fan-out over episodes)

Heavy imports (torch / transformers / the concrete ``AgentTeam``) are deferred
to first request so CI can import this module without GPU or model weights.
"""

from __future__ import annotations

import asyncio
import bisect
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException

from .cache import build_cache_from_env, make_key
from .schemas import (
    AgentStep,
    BenchmarkRunRequest,
    BenchmarkRunResponse,
    HealthResponse,
    MetricsResponse,
    RunEpisodeRequest,
    RunEpisodeResponse,
)

logger = logging.getLogger("lat_agent_team.serve")
logging.basicConfig(level=os.getenv("LAT_LOG_LEVEL", "INFO"))


class _TeamHandle:
    def __init__(self) -> None:
        self._team: Any | None = None
        self._cfg: Any | None = None
        self._episodes_total = 0
        self._successes = 0
        self._latent_bits_total = 0
        self._episode_latencies_ms: list[float] = []

    def ensure_started(self) -> Any:
        if self._team is not None:
            return self._team

        from latent_agent_team.team import AgentTeam  # type: ignore
        from omegaconf import OmegaConf  # type: ignore

        cfg_path = os.getenv("LAT_SERVE_CONFIG", "configs/phi3.yaml")
        cfg = OmegaConf.load(cfg_path)
        self._cfg = cfg
        team = AgentTeam.from_config(cfg)
        self._team = team
        logger.info("AgentTeam ready with backbone=%s", getattr(cfg, "backbone", "?"))
        return team

    def record(self, success: bool, latent_bits: int, latency_ms: float) -> None:
        self._episodes_total += 1
        if success:
            self._successes += 1
        self._latent_bits_total += latent_bits
        if len(self._episode_latencies_ms) >= 1024:
            self._episode_latencies_ms.pop(0)
        bisect.insort(self._episode_latencies_ms, latency_ms)

    def percentile(self, p: float) -> float:
        if not self._episode_latencies_ms:
            return 0.0
        idx = min(len(self._episode_latencies_ms) - 1,
                  int(len(self._episode_latencies_ms) * p))
        return self._episode_latencies_ms[idx]

    @property
    def stats(self) -> tuple[int, int, float]:
        avg_bits = (self._latent_bits_total / self._episodes_total
                    if self._episodes_total else 0.0)
        return self._episodes_total, self._successes, avg_bits


handle = _TeamHandle()
cache = build_cache_from_env()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("latent-agent-team serve starting (cache=%s)", cache.backend_name)
    yield
    logger.info("latent-agent-team serve shutting down")


app = FastAPI(
    title="Latent Agent Team API",
    description="5-agent orchestrator with VQ / continuous latent communication.",
    version="0.1.0",
    lifespan=lifespan,
)


def _episode_to_response(episode_id: str, result: Any,
                         comm_mode: str, latency_ms: float) -> RunEpisodeResponse:
    steps = [
        AgentStep(
            step=i,
            agent=getattr(s, "agent", "planner"),
            action=getattr(s, "action", ""),
            latent_bits=getattr(s, "latent_bits", None),
            audit=getattr(s, "audit", None),
        )
        for i, s in enumerate(getattr(result, "steps", []))
    ]
    total_bits = int(getattr(result, "total_latent_bits", 0))
    avg_bitrate = (total_bits / len(steps)) if steps else 0.0
    return RunEpisodeResponse(
        episode_id=episode_id,
        success=bool(getattr(result, "success", False)),
        final_answer=getattr(result, "final_answer", None),
        steps=steps,
        total_latent_bits=total_bits,
        average_bitrate=avg_bitrate,
        latency_ms=latency_ms,
        comm_mode=comm_mode,  # type: ignore
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    ready = handle._team is not None
    backbone = None
    if ready and handle._cfg is not None:
        backbone = str(getattr(handle._cfg, "backbone", "?"))
    return HealthResponse(
        status="ok" if ready else "degraded",
        team_ready=ready,
        backbone=backbone,
        comm_modes_available=["continuous", "vq", "text"],
        cache_backend=cache.backend_name,
    )


@app.get("/metrics", response_model=MetricsResponse)
async def metrics() -> MetricsResponse:
    total, succ, avg_bits = handle.stats
    return MetricsResponse(
        episodes_total=total,
        success_total=succ,
        success_rate=(succ / total) if total else 0.0,
        avg_bitrate=avg_bits,
        p95_episode_latency_ms=handle.percentile(0.95),
    )


@app.post("/v1/run_episode", response_model=RunEpisodeResponse)
async def run_episode(req: RunEpisodeRequest) -> RunEpisodeResponse:
    key = make_key(req.instruction, req.env_spec, req.comm_mode, req.bitrate_k, req.seed)
    cached = cache.get(key)
    if cached is not None:
        return RunEpisodeResponse(**json.loads(cached))

    try:
        team = handle.ensure_started()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"team unavailable: {exc}")

    t0 = time.perf_counter()
    try:
        result = team.run_episode(
            instruction=req.instruction,
            env=req.env_spec,
            max_steps=req.max_steps,
            comm_mode=req.comm_mode,
            bitrate_k=req.bitrate_k,
            seed=req.seed,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"episode failed: {exc}")

    latency_ms = (time.perf_counter() - t0) * 1000.0
    resp = _episode_to_response(str(uuid.uuid4()), result, req.comm_mode, latency_ms)
    handle.record(resp.success, resp.total_latent_bits, latency_ms)
    cache.set(key, resp.model_dump_json())
    return resp


@app.post("/v1/run_benchmark", response_model=BenchmarkRunResponse)
async def run_benchmark(req: BenchmarkRunRequest) -> BenchmarkRunResponse:
    try:
        team = handle.ensure_started()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"team unavailable: {exc}")

    # Lazy import benchmark wrapper to avoid pulling httpx et al at module-load
    try:
        if req.benchmark == "mind2web":
            from latent_agent_team.benchmarks.mind2web_wrapper import load_examples  # type: ignore
            metric_name = "element_accuracy"
        elif req.benchmark == "webshop":
            from latent_agent_team.benchmarks.webshop_wrapper import load_examples  # type: ignore
            metric_name = "success_rate"
        else:
            from latent_agent_team.benchmarks.agentbench_wrapper import load_examples  # type: ignore
            metric_name = "success_rate"
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"benchmark load failed: {exc}")

    examples = load_examples(split=req.split, max_examples=req.max_examples)

    sem = asyncio.Semaphore(req.concurrency)
    t0 = time.perf_counter()

    async def _one(ex: dict[str, Any]) -> RunEpisodeResponse:
        async with sem:
            inner = RunEpisodeRequest(
                instruction=ex["instruction"],
                env_spec=ex.get("env_spec", {}),
                max_steps=ex.get("max_steps", 15),
                comm_mode=req.comm_mode,
                seed=ex.get("seed"),
            )
            return await run_episode(inner)

    results = await asyncio.gather(*[_one(ex) for ex in examples])
    wall = time.perf_counter() - t0

    success = sum(1 for r in results if r.success)
    n = len(results)
    total_bits = sum(r.total_latent_bits for r in results)
    total_steps = sum(len(r.steps) for r in results) or 1

    return BenchmarkRunResponse(
        benchmark=req.benchmark,
        n_examples=n,
        metric_name=metric_name,
        metric_value=success / n if n else 0.0,
        success_rate=success / n if n else 0.0,
        avg_latent_bits_per_step=total_bits / total_steps,
        wall_time_seconds=wall,
    )
