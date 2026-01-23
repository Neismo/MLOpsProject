#!/usr/bin/env python3
"""Load test script for the FAISS search API."""

import argparse
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm

API_URL = "https://api-94748064668.europe-north2.run.app/search"

SAMPLE_ABSTRACTS = [
    "Machine learning techniques for natural language processing and text classification.",
    "Deep neural networks for computer vision and image recognition tasks.",
    "Reinforcement learning algorithms for robotics and autonomous systems.",
    "Graph neural networks for molecular property prediction in drug discovery.",
    "Transformer architectures for sequence-to-sequence modeling and translation.",
    "Federated learning for privacy-preserving distributed machine learning.",
    "Attention mechanisms in neural networks for improved model interpretability.",
    "Generative adversarial networks for image synthesis and data augmentation.",
    "Transfer learning approaches for low-resource natural language understanding.",
    "Bayesian optimization for hyperparameter tuning in deep learning models.",
]


def make_request(worker_id: int, request_id: int) -> dict:
    """Make a single search request."""
    abstract = SAMPLE_ABSTRACTS[request_id % len(SAMPLE_ABSTRACTS)]
    start = time.perf_counter()
    try:
        response = requests.post(
            API_URL,
            json={"abstract": abstract, "k": 5},
            timeout=30,
        )
        elapsed = time.perf_counter() - start
        return {
            "worker": worker_id,
            "request": request_id,
            "status": response.status_code,
            "latency": elapsed,
            "success": response.status_code == 200,
        }
    except Exception as e:
        elapsed = time.perf_counter() - start
        return {
            "worker": worker_id,
            "request": request_id,
            "status": 0,
            "latency": elapsed,
            "success": False,
            "error": str(e),
        }


def worker_loop(worker_id: int, num_requests: int, progress: tqdm, lock: threading.Lock) -> list[dict]:
    """Run requests for a single worker."""
    results = []
    for i in range(num_requests):
        result = make_request(worker_id, i)
        results.append(result)
        with lock:
            progress.update(1)
    return results


def main():
    parser = argparse.ArgumentParser(description="Load test the FAISS search API")
    parser.add_argument("--workers", type=int, default=10, help="Number of concurrent workers")
    parser.add_argument("--requests", type=int, default=1000, help="Requests per worker")
    args = parser.parse_args()

    total_requests = args.workers * args.requests
    print(f"Starting load test: {args.workers} workers x {args.requests} requests = {total_requests} total")
    print(f"Target: {API_URL}")
    print()

    all_results = []
    start_time = time.perf_counter()
    lock = threading.Lock()

    with tqdm(total=total_requests, desc="Requests", unit="req") as progress:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(worker_loop, worker_id, args.requests, progress, lock): worker_id
                for worker_id in range(args.workers)
            }

            for future in as_completed(futures):
                futures[future]
                results = future.result()
                all_results.extend(results)

    total_time = time.perf_counter() - start_time

    # Calculate stats
    successful = [r for r in all_results if r["success"]]
    failed = [r for r in all_results if not r["success"]]
    latencies = [r["latency"] for r in successful]

    print()
    print("=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"Total requests:    {len(all_results)}")
    print(f"Successful:        {len(successful)} ({100 * len(successful) / len(all_results):.1f}%)")
    print(f"Failed:            {len(failed)}")
    print(f"Total time:        {total_time:.2f}s")
    print(f"Requests/sec:      {len(all_results) / total_time:.2f}")
    print()
    if latencies:
        latencies.sort()
        print(f"Latency min:       {min(latencies) * 1000:.1f}ms")
        print(f"Latency max:       {max(latencies) * 1000:.1f}ms")
        print(f"Latency avg:       {sum(latencies) / len(latencies) * 1000:.1f}ms")
        print(f"Latency p50:       {latencies[len(latencies) // 2] * 1000:.1f}ms")
        print(f"Latency p95:       {latencies[int(len(latencies) * 0.95)] * 1000:.1f}ms")
        print(f"Latency p99:       {latencies[int(len(latencies) * 0.99)] * 1000:.1f}ms")

    if failed:
        print()
        print("Sample errors:")
        for r in failed[:5]:
            print(f"  Worker {r['worker']}, Request {r['request']}: {r.get('error', f'HTTP {r["status"]}')}")


if __name__ == "__main__":
    main()
