"""
Benchmark latency + throughput of the running service.

Usage:
    python scripts/benchmark.py --audio turn_001_user.wav [--url http://localhost:8000] [--n 200] [--concurrency 16]
"""

import argparse
import base64
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def load_b64(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()


def post(url: str, b64: str) -> float:
    body = json.dumps({"audio": b64}).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req) as resp:
        resp.read()
    return time.perf_counter() - t0


def run(url: str, b64: str, n: int, concurrency: int):
    latencies = []
    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(post, url, b64) for _ in range(n)]
        for f in as_completed(futures):
            try:
                latencies.append(f.result())
            except Exception as e:
                print(f"  request failed: {e}")
    elapsed = time.perf_counter() - t_start

    latencies.sort()
    p50 = statistics.median(latencies)
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)]
    rps = len(latencies) / elapsed

    print(f"n={len(latencies)}  concurrency={concurrency}  elapsed={elapsed:.2f}s")
    print(f"p50={p50 * 1000:.1f}ms  p95={p95 * 1000:.1f}ms  p99={p99 * 1000:.1f}ms")
    print(f"throughput={rps:.1f} req/s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--url", default="http://localhost:8000/v1/gender/predict")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=16)
    args = parser.parse_args()

    b64 = load_b64(args.audio)
    print(f"Warming up (5 requests) → {args.url}")
    for _ in range(5):
        post(args.url, b64)

    print(f"Benchmarking {args.n} requests at concurrency={args.concurrency} ...")
    run(args.url, b64, args.n, args.concurrency)


if __name__ == "__main__":
    main()
