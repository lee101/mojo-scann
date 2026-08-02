# mojo-scann

`mojo-scann` is a standalone Mojo port of the dense, learned-quantization path
in [ScaNN](https://github.com/google-research/google-research/tree/master/scann).
It keeps ScaNN's familiar native Python builder API while moving the two hot
inner loops into a single Mojo shared library: asymmetric product-quantization
(AH) lookup scoring and exact L2/dot-product reordering.

The covered subset is intentionally narrow but practical:

- `scann.scann_ops_pybind.builder(dataset, num_neighbors, distance_measure)`
- `.score_ah(...).reorder(...).build()` for learned product quantization
- `.score_brute_force()`, `.tree(...)`, `.search(...)`, `.search_batched(...)`,
  and `.search_batched_parallel(...)`
- `squared_l2`, `dot_product`, and `cosine`, including optional integer `docids`

It does not implement TensorFlow ops, serialized searchers, SOAR, anisotropic
training, LUT16 packing, multi-threaded tree traversal, or ScaNN's full tuning
and persistence API. Tree configuration partitions the candidate set before
ranking, while the implementation deliberately keeps PQ scoring flat and
simple. This is a focused port, not a replacement for every ScaNN deployment.

## Install and use

```bash
pixi install
pixi run build
```

`python/` is activated automatically by Pixi. The example below runs unchanged
inside `pixi run python`:

```python
import numpy as np
import scann

rng = np.random.default_rng(0)
database = rng.normal(size=(10_000, 64))
queries = rng.normal(size=(3, 64))

searcher = (scann.scann_ops_pybind.builder(database, 10, "squared_l2")
            .score_ah(4, min_cluster_size=100)
            .reorder(100)
            .build())
neighbors, squared_distances = searcher.search_batched(queries)
print(neighbors.shape, squared_distances.shape)  # (3, 10) (3, 10)
```

## How it works

Training uses deterministic NumPy Lloyd k-means independently for each padded
subvector. A database vector is then an `int64[n_vectors, n_blocks]` code array;
centroids are contiguous `float64[n_blocks, n_centroids, block_width]`. For a
query, Python forms a small `n_blocks × n_centroids` lookup table. Mojo scans
the code array and sums one table entry per block, avoiding reconstruction of
every database vector. The top approximate candidates can then be scored
exactly by a Mojo L2 or dot-product kernel.

The C ABI passes NumPy buffers as `Int` addresses, recreating typed
`UnsafePointer`s inside the one Mojo compilation unit. `ctypes` owns all memory:
the library never allocates or retains a Python buffer.

## Validation

Run the suite with:

```bash
pixi run test
```

The tests assert exact brute-force L2, dot and cosine parity against NumPy;
verify that reordered PQ scores are exact for their returned ids; test tree,
docid and batched-search behavior; and launch the installed upstream `scann`
wheel in a clean interpreter for direct `score_brute_force` result parity.

## Benchmarks

The command is `pixi run bench`; each reported time is the best of three
searches and includes query scoring plus reordering, not PQ training.

| kernel | mojo-scann | upstream scann | ratio | result |
| --- | ---: | ---: | ---: | --- |
| PQ search + reorder (20k x 32, 100 queries) | 34.4 ms | 2.7 ms | 0.08x | slower |

The CPU path uses SIMD reductions in the Mojo exact-score kernels and batches
lookup scoring, shortlist selection, and reordering to avoid per-query ctypes
overhead. This port is CPU-only.

## Development

```bash
pixi run build && pixi run test && pixi run bench
```

The build writes `dist/libmojo-scann.so`. `MOJO_SCANN_LIB` may point ctypes at a
prebuilt shared library for deployment.
