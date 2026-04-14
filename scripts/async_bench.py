"""Asynchronous concurrent benchmark runner for latent-agent-team.

Fans out a list of benchmark examples against ``POST /v1/run_episode`` with a
bounded ``asyncio.Semaphore`` so a single inference service can be saturated
by many workers without overshooting its concurrency budget.

Writes one JSONL record per episode and prints final aggregate metrics.

Example::

    python scripts/async_bench.py \\
        --benchmark mind2web \\
        --split test --limit 200 \\
        --endpoint http://localhost:8000/v1/run_episode \\
        --comm-mode vq --concurrency 8
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

try:
    import httpx  # type: ignore
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore


async def _one(
    client: "httpx.AsyncClient",
    endpoint: str,
    instruction: str,
    env_spec: dict,
    max_steps: int,
    comm_mode: str,
    retries: int,
    sem: asyncio.Semaphore,
) -> dict:
    payload = {
        "instruction": instruction,
        "env_spec": env_spec,
        "max_steps": max_steps,
        "comm_mode": comm_mode,
    }
    delay = 1.0
    async with sem:
        for attempt in range(retries + 1):
            try:
                r = await client.post(endpoint, json=payload, timeout=180.0)
                r.raise_for_status()
                return r.json()
            except Exception as exc:
                if attempt == retries:
                    return {"episode_id": "", "success": False, "error": str(exc)}
                await asyncio.sleep(delay)
                delay *= 2


def _load_jsonl(path: Path, limit: int) -> list[dict]:
    out: list[dict] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
            if len(out) >= limit:
                break
    return out


async def _run(args: argparse.Namespace) -> None:
    if httpx is None:
        print("httpx is required: pip install httpx", file=sys.stderr)
        sys.exit(2)

    src = Path(args.input) if args.input else Path(f"data/{args.benchmark}_{args.split}.jsonl")
    if not src.exists():
        print(f"benchmark file not found: {src}", file=sys.stderr)
        sys.exit(1)

    examples = _load_jsonl(src, limit=args.limit)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sem = asyncio.Semaphore(args.concurrency)
    t0 = time.perf_counter()

    async with httpx.AsyncClient() as client:
        tasks = [
            _one(
                client,
                args.endpoint,
                ex["instruction"],
                ex.get("env_spec", {}),
                ex.get("max_steps", 15),
                args.comm_mode,
                args.retries,
                sem,
            )
            for ex in examples
        ]
        results = await asyncio.gather(*tasks)

    elapsed = time.perf_counter() - t0

    n = len(results)
    succ = sum(1 for r in results if r.get("success"))
    total_bits = sum(int(r.get("total_latent_bits", 0)) for r in results)
    avg_latency = sum(float(r.get("latency_ms", 0.0)) for r in results) / max(1, n)

    with out_path.open("w") as fh:
        for ex, r in zip(examples, results):
            fh.write(json.dumps({"example": ex, "result": r}) + "\n")

    print(
        f"benchmark={args.benchmark} split={args.split} comm_mode={args.comm_mode}\n"
        f"  episodes={n} · success={succ} ({succ / max(1, n):.1%})\n"
        f"  latent_bits_total={total_bits} · avg_episode_latency={avg_latency:.0f}ms\n"
        f"  wall={elapsed:.1f}s · throughput={n / max(elapsed, 1e-6):.2f} eps/s"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    parser.add_argument("--benchmark", required=True, choices=["mind2web", "webshop", "agentbench"])
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default="outputs/async_bench_out.jsonl")
    parser.add_argument("--endpoint", default="http://localhost:8000/v1/run_episode")
    parser.add_argument("--comm-mode", default="vq", choices=["continuous", "vq", "text"])
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--retries", type=int, default=2)
    asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    main()
