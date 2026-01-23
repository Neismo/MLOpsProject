#!/usr/bin/env python3
"""Load test script hitting /health endpoint."""

import argparse
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm

API_URL = "https://api-94748064668.europe-north2.run.app/health"


def make_request(worker_id: int, request_id: int) -> dict:
    """Make a single health request."""
    start = time.perf_counter()
    try:
        response = requests.get(API_URL, timeout=10)
        elapsed = time.perf_counter() - start
        return {
            "status": response.status_code,
            "latency": elapsed,
            "success": response.status_code == 200,
        }
    except Exception as e:
        elapsed = time.perf_counter() - start
        return {
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
    parser = argparse.ArgumentParser(description="Load test /health endpoint")
    parser.add_argument("--workers", type=int, default=10, help="Number of concurrent workers")
    parser.add_argument("--requests", type=int, default=1000, help="Requests per worker")
    args = parser.parse_args()

    total_requests = args.workers * args.requests
    print(f"Load test: {args.workers} workers x {args.requests} = {total_requests} total")
    print(f"Target: {API_URL}\n")

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
                all_results.extend(future.result())

    total_time = time.perf_counter() - start_time

    successful = [r for r in all_results if r["success"]]
    failed = [r for r in all_results if not r["success"]]
    latencies = [r["latency"] for r in successful]

    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"Total requests:    {len(all_results)}")
    print(f"Successful:        {len(successful)} ({100 * len(successful) / len(all_results):.1f}%)")
    print(f"Failed:            {len(failed)}")
    print(f"Total time:        {total_time:.2f}s")
    print(f"Requests/sec:      {len(all_results) / total_time:.2f}")

    if latencies:
        latencies.sort()
        print(f"\nLatency min:       {min(latencies) * 1000:.1f}ms")
        print(f"Latency avg:       {sum(latencies) / len(latencies) * 1000:.1f}ms")
        print(f"Latency p50:       {latencies[len(latencies) // 2] * 1000:.1f}ms")
        print(f"Latency p95:       {latencies[int(len(latencies) * 0.95)] * 1000:.1f}ms")
        print(f"Latency p99:       {latencies[int(len(latencies) * 0.99)] * 1000:.1f}ms")
        print(f"Latency max:       {max(latencies) * 1000:.1f}ms")


if __name__ == "__main__":
    main()
