"""Numerical and behavioural checks against NumPy and the upstream ScaNN wheel."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

import scann


@pytest.fixture(scope="module")
def vectors():
    rng = np.random.default_rng(42)
    data = np.ascontiguousarray(rng.normal(size=(320, 12)))
    queries = np.ascontiguousarray(rng.normal(size=(9, 12)))
    return data, queries


def test_brute_force_matches_numpy_exactly(vectors):
    data, queries = vectors
    searcher = scann.scann_ops_pybind.builder(data, 7, "squared_l2").score_brute_force().build()
    for query in queries:
        ids, distances = searcher.search(query, final_num_neighbors=7)
        reference = ((data - query) ** 2).sum(axis=1)
        expected = np.argsort(reference, kind="stable")[:7]
        assert np.array_equal(ids, expected)
        assert np.allclose(distances, reference[expected], rtol=1e-6, atol=1e-6)
    ids, _ = searcher.search(queries[0])
    assert ids.shape == (7,)


def test_dot_and_cosine_match_numpy(vectors):
    data, queries = vectors
    dot = scann.builder(data, 5, "dot_product").score_brute_force().build()
    ids, scores = dot.search(queries[0], 5)
    expected = np.argsort(-(data @ queries[0]), kind="stable")[:5]
    assert np.array_equal(ids, expected)
    assert np.allclose(scores, data[expected] @ queries[0], rtol=1e-6)

    cosine = scann.builder(data, 5, "cosine").score_brute_force().build()
    ids, scores = cosine.search(queries[1], 5)
    norm_data = data / np.linalg.norm(data, axis=1, keepdims=True)
    norm_query = queries[1] / np.linalg.norm(queries[1])
    expected = np.argsort(-(norm_data @ norm_query), kind="stable")[:5]
    assert np.array_equal(ids, expected)
    assert np.allclose(scores, norm_data[expected] @ norm_query, rtol=1e-6)


def test_asymmetric_hash_reordering_returns_exact_final_scores(vectors):
    data, queries = vectors
    searcher = (scann.builder(data, 8, "squared_l2")
                .score_ah(3, min_cluster_size=8, training_iterations=12)
                .reorder(80).build())
    ids, distances = searcher.search(queries[0], 8)
    reference = ((data[ids] - queries[0]) ** 2).sum(axis=1)
    assert len(np.unique(ids)) == 8
    assert np.allclose(distances, reference, rtol=1e-6, atol=1e-6)
    exact = np.argsort(((data - queries[0]) ** 2).sum(axis=1))[:8]
    assert len(set(ids) & set(exact)) >= 6


def test_batched_asymmetric_hash_matches_single_query_search(vectors):
    data, queries = vectors
    searcher = (scann.builder(data, 6, "squared_l2")
                .score_ah(3, min_cluster_size=8, training_iterations=12)
                .reorder(64).build())
    ids, distances = searcher.search_batched_parallel(queries[:4], 6)
    expected = [searcher.search(query, 6) for query in queries[:4]]
    assert np.array_equal(ids, np.stack([pair[0] for pair in expected]))
    assert np.allclose(distances, np.stack([pair[1] for pair in expected]), rtol=1e-6, atol=1e-6)


def test_tree_restricts_candidates_and_batched_has_search_shape(vectors):
    data, queries = vectors
    searcher = (scann.builder(data, 4, "squared_l2")
                .tree(8, 3, training_iterations=8)
                .score_ah(2, min_cluster_size=8)
                .reorder(32).build(docids=np.arange(1000, 1000 + len(data))))
    ids, distances = searcher.search_batched(queries[:3], 4)
    assert ids.shape == distances.shape == (3, 4)
    assert np.all((1000 <= ids) & (ids < 1000 + len(data)))
    assert np.all(np.diff(distances, axis=1) >= 0)


@pytest.mark.parametrize("rows, queries", [(257, 3), (2048, 40)])
def test_batched_l2_handles_simd_tails_and_parallel_threshold(rows, queries):
    rng = np.random.default_rng(rows + queries)
    data = np.ascontiguousarray(rng.normal(size=(rows, 7)))
    query_array = np.ascontiguousarray(rng.normal(size=(queries, 7)))
    searcher = scann.builder(data, 4, "squared_l2").score_brute_force().build()
    ids, distances = searcher.search_batched(query_array, 4)
    for row, actual_ids, actual_distances in zip(query_array, ids, distances):
        reference = ((data - row) ** 2).sum(axis=1)
        expected = np.argsort(reference, kind="stable")[:4]
        assert np.array_equal(actual_ids, expected)
        assert np.allclose(actual_distances, reference[expected], rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("rows, queries", [(257, 3), (2048, 40)])
def test_fused_pq_rerank_handles_simd_tails_and_parallel_threshold(rows, queries):
    rng = np.random.default_rng(2 * rows + queries)
    data = np.ascontiguousarray(rng.normal(size=(rows, 7)))
    query_array = np.ascontiguousarray(rng.normal(size=(queries, 7)))
    searcher = (scann.builder(data, 4, "squared_l2")
                .score_ah(3, min_cluster_size=64, training_iterations=4)
                .reorder(32).build())
    ids, distances = searcher.search_batched(query_array, 4)
    expected = [searcher.search(query, 4) for query in query_array]
    assert np.array_equal(ids, np.stack([pair[0] for pair in expected]))
    assert np.allclose(distances, np.stack([pair[1] for pair in expected]),
                       rtol=1e-6, atol=1e-6)


def test_upstream_wheel_brute_force_parity(vectors):
    """Run installed upstream outside this checkout, avoiding our matching package name."""
    data, queries = vectors
    payload = {"data": data.tolist(), "query": queries[0].tolist()}
    script = """
import json, sys
import numpy as np
import scann
p = json.loads(sys.stdin.read())
x = np.asarray(p['data'], dtype=np.float32)
q = np.asarray(p['query'], dtype=np.float32)
s = scann.scann_ops_pybind.builder(x, 6, 'squared_l2').score_brute_force().build()
ids, dist = s.search(q, final_num_neighbors=6)
print(json.dumps([ids.tolist(), np.asarray(dist).tolist()]))
"""
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    proc = subprocess.run([sys.executable, "-c", script], cwd="/tmp", input=json.dumps(payload),
                          text=True, capture_output=True, env=env, timeout=120)
    assert proc.returncode == 0, proc.stderr
    upstream_ids, upstream_distances = json.loads(proc.stdout)
    ours = scann.builder(data.astype(np.float32), 6, "squared_l2").score_brute_force().build()
    ours_ids, ours_distances = ours.search(queries[0].astype(np.float32), 6)
    assert np.array_equal(ours_ids, upstream_ids)
    assert np.allclose(ours_distances, upstream_distances, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("distance", ["bad", "euclidean"])
def test_invalid_distance_is_rejected(vectors, distance):
    with pytest.raises(ValueError):
        scann.builder(vectors[0], 3, distance)


@pytest.mark.parametrize("dtype", [np.int64, np.complex128])
def test_ffi_vector_boundary_rejects_unsafe_dtypes(vectors, dtype):
    with pytest.raises(TypeError, match="float32 or float64"):
        scann.builder(vectors[0].astype(dtype), 3, "squared_l2")


def test_ffi_boundary_accepts_strided_float32_and_rejects_empty_batches(vectors):
    data, queries = vectors
    searcher = scann.builder(data.astype(np.float32)[:, ::-1], 3, "squared_l2").score_brute_force().build()
    ids, _ = searcher.search(queries[0, ::-1].astype(np.float32), 3)
    assert ids.shape == (3,)
    with pytest.raises(ValueError, match="at least one"):
        searcher.search_batched(np.empty((0, data.shape[1]), dtype=np.float64))


def test_docids_reject_lossy_integer_conversion(vectors):
    builder = scann.builder(vectors[0], 3, "squared_l2").score_brute_force()
    with pytest.raises(OverflowError, match="fit in int64"):
        builder.build(docids=np.full(len(vectors[0]), np.iinfo(np.uint64).max, dtype=np.uint64))
