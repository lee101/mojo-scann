"""Hot loops for product-quantized vector search.

The Python layer owns arrays and training.  This unit only receives raw addresses,
which keeps the exported ABI concrete and makes scoring usable from ctypes.
"""

from std.sys.info import simd_width_of

comptime FPtr = Pointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = Pointer[Int64, AnyOrigin[mut=True]]
comptime W = simd_width_of[DType.float64]()


@export("msc_l2_scores")
def msc_l2_scores(data: Int, query: Int, dst: Int, n: Int, d: Int) abi("C"):
    var x = FPtr(unsafe_from_address=data)
    var q = FPtr(unsafe_from_address=query)
    var result = FPtr(unsafe_from_address=dst)
    for row in range(n):
        var total = 0.0
        var vector_total = SIMD[DType.float64, W](0)
        var col = 0
        while col + W <= d:
            var delta = x.unsafe_load[width=W](row * d + col) - q.unsafe_load[width=W](col)
            vector_total += delta * delta
            col += W
        total = vector_total.reduce_add()
        while col < d:
            var delta = x[unsafe_offset=row * d + col] - q[unsafe_offset=col]
            total += delta * delta
            col += 1
        result[unsafe_offset=row] = total


@export("msc_dot_scores")
def msc_dot_scores(data: Int, query: Int, dst: Int, n: Int, d: Int) abi("C"):
    var x = FPtr(unsafe_from_address=data)
    var q = FPtr(unsafe_from_address=query)
    var result = FPtr(unsafe_from_address=dst)
    for row in range(n):
        var total = 0.0
        var vector_total = SIMD[DType.float64, W](0)
        var col = 0
        while col + W <= d:
            vector_total += x.unsafe_load[width=W](row * d + col) * q.unsafe_load[width=W](col)
            col += W
        total = vector_total.reduce_add()
        while col < d:
            total += x[unsafe_offset=row * d + col] * q[unsafe_offset=col]
            col += 1
        result[unsafe_offset=row] = total


@export("msc_l2_scores_batched")
def msc_l2_scores_batched(data: Int, queries: Int, dst: Int, n: Int, d: Int, m: Int) abi("C"):
    var x = FPtr(unsafe_from_address=data)
    var q = FPtr(unsafe_from_address=queries)
    var result = FPtr(unsafe_from_address=dst)
    def score_query(query_row: Int) capturing:
        for row in range(n):
            var total = 0.0
            var vector_total = SIMD[DType.float64, W](0)
            var col = 0
            while col + W <= d:
                var delta = x.unsafe_load[width=W](row * d + col) - q.unsafe_load[width=W](query_row * d + col)
                vector_total += delta * delta
                col += W
            total = vector_total.reduce_add()
            while col < d:
                var delta = x[unsafe_offset=row * d + col] - q[unsafe_offset=query_row * d + col]
                total += delta * delta
                col += 1
            result[unsafe_offset=query_row * n + row] = total
    for query_row in range(m):
        score_query(query_row)


@export("msc_dot_scores_batched")
def msc_dot_scores_batched(data: Int, queries: Int, dst: Int, n: Int, d: Int, m: Int) abi("C"):
    var x = FPtr(unsafe_from_address=data)
    var q = FPtr(unsafe_from_address=queries)
    var result = FPtr(unsafe_from_address=dst)
    def score_query(query_row: Int) capturing:
        for row in range(n):
            var total = 0.0
            var vector_total = SIMD[DType.float64, W](0)
            var col = 0
            while col + W <= d:
                vector_total += x.unsafe_load[width=W](row * d + col) * q.unsafe_load[width=W](query_row * d + col)
                col += W
            total = vector_total.reduce_add()
            while col < d:
                total += x[unsafe_offset=row * d + col] * q[unsafe_offset=query_row * d + col]
                col += 1
            result[unsafe_offset=query_row * n + row] = total
    for query_row in range(m):
        score_query(query_row)


@export("msc_ah_scores")
def msc_ah_scores(codes: Int, lookup: Int, dst: Int, n: Int, blocks: Int, centroids: Int) abi("C"):
    var code = IPtr(unsafe_from_address=codes)
    var table = FPtr(unsafe_from_address=lookup)
    var result = FPtr(unsafe_from_address=dst)
    for row in range(n):
        var total = 0.0
        for block in range(blocks):
            total += table[unsafe_offset=block * centroids + Int(code[unsafe_offset=row * blocks + block])]
        result[unsafe_offset=row] = total


@export("msc_ah_scores_batched")
def msc_ah_scores_batched(codes: Int, lookups: Int, dst: Int, n: Int, blocks: Int, centroids: Int, m: Int) abi("C"):
    var code = IPtr(unsafe_from_address=codes)
    var table = FPtr(unsafe_from_address=lookups)
    var result = FPtr(unsafe_from_address=dst)
    var table_size = blocks * centroids
    def score_query(query_row: Int) capturing:
        for row in range(n):
            var total = 0.0
            for block in range(blocks):
                total += table[unsafe_offset=query_row * table_size + block * centroids + Int(code[unsafe_offset=row * blocks + block])]
            result[unsafe_offset=query_row * n + row] = total
    for query_row in range(m):
        score_query(query_row)
