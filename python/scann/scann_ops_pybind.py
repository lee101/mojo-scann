"""A focused, API-compatible ScaNN builder backed by Mojo PQ scoring.

This covers the native builder/searcher workflow for dense float vectors.  The
configuration surface intentionally accepts ScaNN's tuning arguments even where
this compact implementation does not need every one of them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from . import _lib


def _validate_data(dataset: object) -> np.ndarray:
    data = _lib.f64(dataset)
    if data.ndim != 2 or not data.shape[0] or not data.shape[1]:
        raise ValueError("dataset must be a non-empty two-dimensional array")
    if not np.isfinite(data).all():
        raise ValueError("dataset must contain only finite values")
    return data


def _distance_kind(distance_measure: str) -> str:
    key = distance_measure.lower().replace("-", "_")
    aliases = {"squared_l2": "l2", "l2": "l2", "dot_product": "dot", "cosine": "cosine"}
    if key not in aliases:
        raise ValueError("distance_measure must be 'squared_l2', 'dot_product', or 'cosine'")
    return aliases[key]


def _kmeans(data: np.ndarray, k: int, iterations: int, seed: int = 0) -> np.ndarray:
    """Deterministic Lloyd training, with empty clusters kept at their seed."""
    k = min(k, len(data))
    rng = np.random.default_rng(seed)
    centers = data[rng.choice(len(data), size=k, replace=False)].copy()
    for _ in range(max(1, iterations)):
        labels = _nearest(data, centers)
        next_centers = centers.copy()
        for cluster in range(k):
            members = data[labels == cluster]
            if len(members):
                next_centers[cluster] = members.mean(axis=0)
        if np.allclose(next_centers, centers, rtol=0.0, atol=1e-12):
            break
        centers = next_centers
    return centers


def _nearest(data: np.ndarray, centers: np.ndarray, chunk: int = 2048) -> np.ndarray:
    """Nearest centroid labels without materialising an N-by-K training matrix."""
    labels = np.empty(len(data), dtype=np.int64)
    for start in range(0, len(data), chunk):
        section = data[start:start + chunk]
        distances = ((section[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels[start:start + len(section)] = distances.argmin(axis=1)
    return labels


def _top(scores: np.ndarray, count: int, descending: bool) -> np.ndarray:
    count = min(max(1, int(count)), len(scores))
    if count == len(scores):
        selected = np.arange(len(scores))
    elif descending:
        selected = np.argpartition(scores, len(scores) - count)[-count:]
    else:
        selected = np.argpartition(scores, count - 1)[:count]
    order = np.lexsort((selected, -scores[selected] if descending else scores[selected]))
    return selected[order]


def _top_batched(scores: np.ndarray, count: int, descending: bool) -> np.ndarray:
    count = min(max(1, int(count)), scores.shape[1])
    if count == scores.shape[1]:
        selected = np.broadcast_to(np.arange(count), scores.shape)
    elif descending:
        selected = np.argpartition(scores, scores.shape[1] - count, axis=1)[:, -count:]
    else:
        selected = np.argpartition(scores, count - 1, axis=1)[:, :count]
    rows = np.arange(len(scores))[:, None]
    values = scores[rows, selected]
    order = np.lexsort((selected, -values if descending else values), axis=1)
    return np.take_along_axis(selected, order, axis=1)


@dataclass
class _PQ:
    centers: np.ndarray
    codes: np.ndarray
    width: int


class ScannSearcher:
    """An in-memory dense-vector searcher returned by :class:`ScannBuilder`."""

    def __init__(self, data: np.ndarray, distance: str, pq: _PQ | None, reorder: int,
                 leaf_ids: np.ndarray | None = None, leaf_centers: np.ndarray | None = None,
                 default_leaves: int | None = None, docids: np.ndarray | None = None,
                 num_neighbors: int = 10):
        self._data = data
        self._distance = distance
        self._pq = pq
        self._reorder = reorder
        self._leaf_ids = leaf_ids
        self._leaf_centers = leaf_centers
        self._default_leaves = default_leaves
        self._docids = docids
        self._num_neighbors = num_neighbors
        self._descending = distance in ("dot", "cosine")

    @property
    def size(self) -> int:
        return len(self._data)

    def _query(self, query: object) -> np.ndarray:
        result = _lib.f64(query)
        if result.ndim != 1 or len(result) != self._data.shape[1]:
            raise ValueError(f"query must have shape ({self._data.shape[1]},)")
        if not np.isfinite(result).all():
            raise ValueError("query must contain only finite values")
        if self._distance == "cosine":
            norm = np.linalg.norm(result)
            if norm:
                result = result / norm
        return result

    def _lookup(self, query: np.ndarray) -> np.ndarray:
        assert self._pq is not None
        centers, width = self._pq.centers, self._pq.width
        padded = np.pad(query, (0, centers.shape[0] * width - len(query)))
        blocks = padded.reshape(centers.shape[0], width)
        if self._descending:
            return np.einsum("bcd,bd->bc", centers, blocks, optimize=True).astype(np.float64)
        return ((centers - blocks[:, None, :]) ** 2).sum(axis=2)

    def _lookup_batched(self, queries: np.ndarray) -> np.ndarray:
        assert self._pq is not None
        centers, width = self._pq.centers, self._pq.width
        padded_width = centers.shape[0] * width
        if queries.shape[1] == padded_width:
            blocks = queries.reshape(len(queries), centers.shape[0], width)
        else:
            padded = np.zeros((len(queries), padded_width), dtype=np.float64)
            padded[:, :queries.shape[1]] = queries
            blocks = padded.reshape(len(queries), centers.shape[0], width)
        if self._descending:
            return np.einsum("bcd,mbd->mbc", centers, blocks, optimize=True).astype(np.float64)
        return ((centers[None, :, :, :] - blocks[:, :, None, :]) ** 2).sum(axis=3)

    def _scores(self, query: np.ndarray) -> np.ndarray:
        scores = np.empty(self.size, dtype=np.float64)
        api = _lib.lib()
        if self._pq is None:
            fn = api.msc_dot_scores if self._descending else api.msc_l2_scores
            fn(_lib.addr(self._data), _lib.addr(query), _lib.addr(scores), self.size, self._data.shape[1])
        else:
            lookup = np.ascontiguousarray(self._lookup(query), dtype=np.float64)
            api.msc_ah_scores(_lib.addr(self._pq.codes), _lib.addr(lookup), _lib.addr(scores),
                              self.size, self._pq.codes.shape[1], lookup.shape[1])
        return scores

    def _scores_batched(self, queries: np.ndarray) -> np.ndarray:
        scores = np.empty((len(queries), self.size), dtype=np.float64)
        api = _lib.lib()
        if self._pq is None:
            fn = api.msc_dot_scores_batched if self._descending else api.msc_l2_scores_batched
            fn(_lib.addr(self._data), _lib.addr(queries), _lib.addr(scores), self.size,
               self._data.shape[1], len(queries))
        else:
            lookup = np.ascontiguousarray(self._lookup_batched(queries), dtype=np.float64)
            api.msc_ah_scores_batched(_lib.addr(self._pq.codes), _lib.addr(lookup), _lib.addr(scores),
                                      self.size, self._pq.codes.shape[1], lookup.shape[2], len(queries))
        return scores

    def _candidate_ids(self, query: np.ndarray, leaves_to_search: int) -> np.ndarray:
        if self._leaf_ids is None:
            return np.arange(self.size)
        leaves = self._default_leaves if leaves_to_search < 0 else leaves_to_search
        if leaves is None or leaves <= 0 or leaves >= len(self._leaf_centers):
            return np.arange(self.size)
        delta = self._leaf_centers - query
        chosen = _top((delta * delta).sum(axis=1), leaves, False)
        return np.flatnonzero(np.isin(self._leaf_ids, chosen))

    def _exact(self, query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        if len(candidates) == self.size:
            full = np.empty(self.size, dtype=np.float64)
            fn = _lib.lib().msc_dot_scores if self._descending else _lib.lib().msc_l2_scores
            fn(_lib.addr(self._data), _lib.addr(query), _lib.addr(full), self.size, self._data.shape[1])
            return full
        x = self._data[candidates]
        if self._descending:
            return x @ query
        return ((x - query) ** 2).sum(axis=1)

    def _finish_search(self, q: np.ndarray, approximate: np.ndarray, final_num_neighbors: int,
                       pre_reorder_num_neighbors: int, leaves_to_search: int):
        requested = final_num_neighbors if final_num_neighbors > 0 else self._num_neighbors
        candidates = self._candidate_ids(q, leaves_to_search)
        approximate = approximate[candidates]
        budget = pre_reorder_num_neighbors if pre_reorder_num_neighbors > 0 else self._reorder
        budget = max(requested, budget) if budget else requested
        shortlist = _top(approximate, budget, self._descending)
        candidate_shortlist = candidates[shortlist]
        if self._pq is not None and self._reorder > 0:
            exact = self._exact(q, candidate_shortlist)
            local = _top(exact, requested, self._descending)
            ids, scores = candidate_shortlist[local], exact[local]
        else:
            local = _top(approximate, requested, self._descending)
            ids, scores = candidates[local], approximate[local]
        if self._docids is not None:
            ids = self._docids[ids]
        return ids.astype(np.int64, copy=False), scores.astype(np.float32)

    def _finish_searches_batched(self, queries: np.ndarray, approximate: np.ndarray,
                                 final_num_neighbors: int, pre_reorder_num_neighbors: int):
        requested = final_num_neighbors if final_num_neighbors > 0 else self._num_neighbors
        budget = pre_reorder_num_neighbors if pre_reorder_num_neighbors > 0 else self._reorder
        budget = max(requested, budget) if budget else requested
        shortlist = _top_batched(approximate, budget, self._descending)
        rows = np.arange(len(queries))[:, None]
        if self._pq is not None and self._reorder > 0:
            shortlisted = self._data[shortlist]
            if self._descending:
                exact = np.einsum("mnd,md->mn", shortlisted, queries, optimize=True)
            else:
                delta = shortlisted - queries[:, None, :]
                exact = np.einsum("mnd,mnd->mn", delta, delta, optimize=True)
            local = _top_batched(exact, requested, self._descending)
            ids, scores = shortlist[rows, local], exact[rows, local]
        else:
            ids, scores = shortlist[:, :requested], approximate[rows, shortlist[:, :requested]]
        if self._docids is not None:
            ids = self._docids[ids]
        return ids.astype(np.int64, copy=False), scores.astype(np.float32)

    def search(self, query: object, final_num_neighbors: int = -1,
               pre_reorder_num_neighbors: int = -1, leaves_to_search: int = -1):
        """Return `(neighbors, distances)` with ScaNN's native argument names."""
        q = self._query(query)
        return self._finish_search(q, self._scores(q), final_num_neighbors,
                                   pre_reorder_num_neighbors, leaves_to_search)

    def search_batched(self, queries: object, final_num_neighbors: int = -1,
                       pre_reorder_num_neighbors: int = -1, leaves_to_search: int = -1):
        query_array = _lib.f64(queries)
        if query_array.ndim != 2 or query_array.shape[1] != self._data.shape[1]:
            raise ValueError(f"queries must have shape (n, {self._data.shape[1]})")
        if not len(query_array):
            raise ValueError("queries must contain at least one vector")
        if not np.isfinite(query_array).all():
            raise ValueError("queries must contain only finite values")
        if self._distance == "cosine":
            norms = np.linalg.norm(query_array, axis=1, keepdims=True)
            query_array = query_array / np.where(norms == 0, 1.0, norms)
        approximate = self._scores_batched(query_array)
        if self._leaf_ids is None:
            return self._finish_searches_batched(query_array, approximate, final_num_neighbors,
                                                 pre_reorder_num_neighbors)
        pairs = [self._finish_search(query, scores, final_num_neighbors,
                                     pre_reorder_num_neighbors, leaves_to_search)
                 for query, scores in zip(query_array, approximate)]
        return np.stack([pair[0] for pair in pairs]), np.stack([pair[1] for pair in pairs])

    search_batched_parallel = search_batched


class ScannBuilder:
    def __init__(self, db: object, num_neighbors: int, distance_measure: str):
        self._db = _validate_data(db)
        self._num_neighbors = int(num_neighbors)
        if self._num_neighbors <= 0:
            raise ValueError("num_neighbors must be positive")
        self._distance = _distance_kind(distance_measure)
        self._pq_args: tuple[int, int, int] | None = None
        self._reorder = 0
        self._tree_args: tuple[int, int, int] | None = None
        self._brute_force = False

    def tree(self, num_leaves: int = 100, num_leaves_to_search: int = 10,
             training_sample_size: int = 100000, min_partition_size: int = 50,
             training_iterations: int = 12, spherical: bool = False,
             quantize_centroids: bool = False, random_init: bool = True,
             soar_lambda: float | None = None, overretrieve_factor: float | None = None,
             distance_measure: str | None = None):
        del min_partition_size, spherical, quantize_centroids, random_init, soar_lambda, overretrieve_factor
        if distance_measure is not None:
            self._distance = _distance_kind(distance_measure)
        self._tree_args = (max(1, int(num_leaves)), max(1, int(num_leaves_to_search)), int(training_iterations))
        return self

    def score_ah(self, dimensions_per_block: int, anisotropic_quantization_threshold: float = 0.2,
                 training_sample_size: int = 100000, min_cluster_size: int = 100,
                 hash_type: str = "lut16", training_iterations: int = 10):
        del anisotropic_quantization_threshold, training_sample_size, hash_type
        if dimensions_per_block <= 0:
            raise ValueError("dimensions_per_block must be positive")
        self._pq_args = (int(dimensions_per_block), max(2, int(min_cluster_size)), int(training_iterations))
        self._brute_force = False
        return self

    def score_brute_force(self, *args, **kwargs):
        del args, kwargs
        self._brute_force = True
        self._pq_args = None
        return self

    def reorder(self, reordering_num_neighbors: int, quantize: bool = False):
        del quantize
        self._reorder = max(0, int(reordering_num_neighbors))
        return self

    def build(self, docids: Iterable[int] | None = None, **kwargs) -> ScannSearcher:
        del kwargs
        data = self._db.copy()
        if self._distance == "cosine":
            norms = np.linalg.norm(data, axis=1, keepdims=True)
            data = data / np.where(norms == 0, 1.0, norms)
        ids = None if docids is None else _lib.i64(list(docids))
        if ids is not None and ids.shape != (len(data),):
            raise ValueError("docids must contain one id per database vector")
        pq = None
        if self._pq_args is not None and not self._brute_force:
            width, min_cluster_size, iterations = self._pq_args
            blocks = (data.shape[1] + width - 1) // width
            padded = np.pad(data, ((0, 0), (0, blocks * width - data.shape[1])))
            cluster_count = min(256, max(2, len(data) // min_cluster_size))
            centers = np.empty((blocks, cluster_count, width), dtype=np.float64)
            codes = np.empty((len(data), blocks), dtype=np.int64)
            for block in range(blocks):
                section = padded[:, block * width:(block + 1) * width]
                centers[block] = _kmeans(section, cluster_count, iterations, block)
                codes[:, block] = _nearest(section, centers[block])
            pq = _PQ(centers, np.ascontiguousarray(codes), width)
        leaf_ids = leaf_centers = None
        default_leaves = None
        if self._tree_args is not None:
            leaves, default_leaves, iterations = self._tree_args
            leaf_centers = _kmeans(data, leaves, iterations)
            leaf_ids = _nearest(data, leaf_centers)
        return ScannSearcher(np.ascontiguousarray(data), self._distance, pq, self._reorder,
                             leaf_ids, leaf_centers, default_leaves, ids, self._num_neighbors)


def builder(db: object, num_neighbors: int, distance_measure: str) -> ScannBuilder:
    return ScannBuilder(db, num_neighbors, distance_measure)
