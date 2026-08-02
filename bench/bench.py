"""Measured Mojo-scann timings against the upstream ScaNN native wheel.

Run only through `pixi run bench`; that task takes the shared machine lock.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import scann  # noqa: E402


def best(fn, repeat: int = 3) -> float:
    value = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        value = min(value, time.perf_counter() - start)
    return value


def upstream_worker(data: np.ndarray, queries: np.ndarray) -> float:
    """Time upstream in a clean interpreter so it cannot import this repo's package."""
    path = ROOT / "bench" / ".upstream-input.npz"
    np.savez(path, data=data.astype(np.float32), queries=queries.astype(np.float32))
    script = """
import sys, time, numpy as np
z = np.load(sys.argv[1])
import scann
s = scann.scann_ops_pybind.builder(z['data'], 10, 'squared_l2').score_ah(4).reorder(100).build()
s.search_batched(z['queries'], final_num_neighbors=10)
times = []
for _ in range(3):
    t = time.perf_counter()
    s.search_batched(z['queries'], final_num_neighbors=10)
    times.append(time.perf_counter() - t)
print(min(times))
"""
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    proc = subprocess.run([sys.executable, "-c", script, str(path)], cwd="/tmp", text=True,
                          capture_output=True, env=env, timeout=300)
    path.unlink(missing_ok=True)
    if proc.returncode:
        raise RuntimeError(proc.stderr)
    return float(proc.stdout.strip().splitlines()[-1])


def main() -> None:
    rng = np.random.default_rng(7)
    data = np.ascontiguousarray(rng.normal(size=(20_000, 32)))
    queries = np.ascontiguousarray(rng.normal(size=(100, 32)))
    ours = (scann.builder(data, 10, "squared_l2").score_ah(4, min_cluster_size=100)
            .reorder(100).build())
    ours.search_batched(queries[:2], 10)
    ours_seconds = best(lambda: ours.search_batched(queries, 10))
    upstream_seconds = upstream_worker(data, queries)
    ratio = upstream_seconds / ours_seconds
    verdict = "faster" if ratio > 1 else "slower"
    print("| kernel | mojo-scann | upstream scann | ratio | result |")
    print("| --- | ---: | ---: | ---: | --- |")
    print(f"| PQ search + reorder (20k x 32, 100 queries) | {ours_seconds * 1e3:.1f} ms | "
          f"{upstream_seconds * 1e3:.1f} ms | {ratio:.2f}x | {verdict} |")


if __name__ == "__main__":
    main()
