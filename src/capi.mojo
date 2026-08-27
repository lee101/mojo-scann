"""Hot loops for product-quantized vector search.

The Python layer owns arrays and training.  This unit only receives raw addresses,
which keeps the exported ABI concrete and makes scoring usable from ctypes.
"""

from std.sys.info import simd_width_of

comptime FPtr = Pointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = Pointer[Int64, AnyOrigin[mut=True]]
comptime U8Ptr = Pointer[UInt8, AnyOrigin[mut=True]]
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


@export("msc_exact_candidates_batched")
def msc_exact_candidates_batched(data: Int, queries: Int, candidate_ids: Int, dst: Int,
                                 m: Int, k: Int, d: Int, dot_product: Int) abi("C"):
    var x = FPtr(unsafe_from_address=data)
    var q = FPtr(unsafe_from_address=queries)
    var ids = IPtr(unsafe_from_address=candidate_ids)
    var result = FPtr(unsafe_from_address=dst)
    for query_row in range(m):
        for candidate in range(k):
            var row = Int(ids[unsafe_offset=query_row * k + candidate])
            var vector_total = SIMD[DType.float64, W](0)
            var col = 0
            if dot_product != 0:
                while col + W <= d:
                    vector_total += x.unsafe_load[width=W](row * d + col) * q.unsafe_load[width=W](query_row * d + col)
                    col += W
                var total = vector_total.reduce_add()
                while col < d:
                    total += x[unsafe_offset=row * d + col] * q[unsafe_offset=query_row * d + col]
                    col += 1
                result[unsafe_offset=query_row * k + candidate] = total
            else:
                while col + W <= d:
                    var delta = x.unsafe_load[width=W](row * d + col) - q.unsafe_load[width=W](query_row * d + col)
                    vector_total += delta * delta
                    col += W
                var total = vector_total.reduce_add()
                while col < d:
                    var delta = x[unsafe_offset=row * d + col] - q[unsafe_offset=query_row * d + col]
                    total += delta * delta
                    col += 1
                result[unsafe_offset=query_row * k + candidate] = total


@export("msc_ah_scores")
def msc_ah_scores(codes: Int, lookup: Int, dst: Int, n: Int, blocks: Int, centroids: Int) abi("C"):
    var code = U8Ptr(unsafe_from_address=codes)
    var table = FPtr(unsafe_from_address=lookup)
    var result = FPtr(unsafe_from_address=dst)
    for row in range(n):
        var total = 0.0
        for block in range(blocks):
            total += table[unsafe_offset=block * centroids + Int(code[unsafe_offset=row * blocks + block])]
        result[unsafe_offset=row] = total


@export("msc_ah_scores_batched")
def msc_ah_scores_batched(codes: Int, lookups: Int, dst: Int, n: Int, blocks: Int, centroids: Int, m: Int) abi("C"):
    var code = U8Ptr(unsafe_from_address=codes)
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


@export("msc_ah_top_batched")
def msc_ah_top_batched(codes: Int, lookups: Int, dst_ids: Int, dst_scores: Int,
                       n: Int, blocks: Int, centroids: Int, m: Int, k: Int,
                       descending: Int) abi("C"):
    var code = U8Ptr(unsafe_from_address=codes)
    var table = FPtr(unsafe_from_address=lookups)
    var ids = IPtr(unsafe_from_address=dst_ids)
    var scores = FPtr(unsafe_from_address=dst_scores)
    var table_size = blocks * centroids
    def is_worse(a_score: Float64, a_id: Int64, b_score: Float64, b_id: Int64) capturing -> Bool:
        if descending != 0:
            return a_score < b_score or (a_score == b_score and a_id > b_id)
        return a_score > b_score or (a_score == b_score and a_id > b_id)

    def row_score(query_row: Int, row: Int) capturing -> Float64:
        var total = 0.0
        for block in range(blocks):
            total += table[unsafe_offset=query_row * table_size + block * centroids + Int(code[unsafe_offset=row * blocks + block])]
        return total

    def sift_down(base: Int, start: Int, count: Int) capturing:
        var root = start
        while True:
            var child = root * 2 + 1
            if child >= count:
                break
            var worst = child
            if child + 1 < count and is_worse(
                    scores[unsafe_offset=base + child + 1], ids[unsafe_offset=base + child + 1],
                    scores[unsafe_offset=base + child], ids[unsafe_offset=base + child]):
                worst = child + 1
            if not is_worse(scores[unsafe_offset=base + worst], ids[unsafe_offset=base + worst],
                            scores[unsafe_offset=base + root], ids[unsafe_offset=base + root]):
                break
            var swap_score = scores[unsafe_offset=base + root]
            var swap_id = ids[unsafe_offset=base + root]
            scores[unsafe_offset=base + root] = scores[unsafe_offset=base + worst]
            ids[unsafe_offset=base + root] = ids[unsafe_offset=base + worst]
            scores[unsafe_offset=base + worst] = swap_score
            ids[unsafe_offset=base + worst] = swap_id
            root = worst

    for query_row in range(m):
        var base = query_row * k
        for row in range(k):
            ids[unsafe_offset=base + row] = Int64(row)
            scores[unsafe_offset=base + row] = row_score(query_row, row)
        var parent = k // 2 - 1
        while parent >= 0:
            sift_down(base, parent, k)
            parent -= 1
        for row in range(k, n):
            var candidate_score = row_score(query_row, row)
            if is_worse(scores[unsafe_offset=base], ids[unsafe_offset=base],
                        candidate_score, Int64(row)):
                scores[unsafe_offset=base] = candidate_score
                ids[unsafe_offset=base] = Int64(row)
                sift_down(base, 0, k)
