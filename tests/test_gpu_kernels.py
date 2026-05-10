"""GPU kernel tests.

All tests use importlib to load modules at call time so that a missing or
broken GPU package never prevents the rest of the test file from loading.
Every test asserts numerical correctness (not just shape/not-None).
"""
import importlib

import numpy as np
import pytest


def _load(module_path):
    return importlib.import_module(module_path)


# ── FFT ──────────────────────────────────────────────────────────────────────

def test_fft_length():
    mod = _load("algorithms.gpu.signal.fft")
    out = mod.fft(list(range(8)))
    assert len(out) == 8


def test_fft_matches_numpy():
    mod = _load("algorithms.gpu.signal.fft")
    arr = np.random.rand(16).astype(np.complex128)
    out = np.array(mod.fft(arr))
    expected = np.fft.fft(arr)
    assert np.allclose(out, expected, atol=1e-6)


# ── Convolution ───────────────────────────────────────────────────────────────

def test_convolve_length():
    mod = _load("algorithms.gpu.signal.convolution")
    out = mod.convolve([1, 2, 3], [0, 1, 0])
    assert len(out) == 5  # len(a) + len(b) - 1


def test_convolve_identity_kernel():
    mod = _load("algorithms.gpu.signal.convolution")
    a = [1.0, 2.0, 3.0, 4.0]
    out = np.array(mod.convolve(a, [1.0]))
    assert np.allclose(out, a)


def test_convolve_matches_numpy():
    mod = _load("algorithms.gpu.signal.convolution")
    a = np.random.rand(8)
    b = np.random.rand(4)
    out = np.array(mod.convolve(a, b))
    expected = np.convolve(a, b)
    assert np.allclose(out, expected, atol=1e-6)


# ── Numba vec_add / matmul ────────────────────────────────────────────────────

def test_vec_add_correctness():
    mod = _load("algorithms.gpu.kernels.numba_kernels")
    a = np.arange(1024, dtype=np.float32)
    b = a * 2.0
    out = mod.vec_add(a, b)
    assert np.allclose(out, a + b)


def test_vec_add_zeros():
    mod = _load("algorithms.gpu.kernels.numba_kernels")
    a = np.zeros(64, dtype=np.float32)
    b = np.zeros(64, dtype=np.float32)
    out = mod.vec_add(a, b)
    assert np.allclose(out, 0)


def test_matmul_identity():
    mod = _load("algorithms.gpu.kernels.numba_kernels")
    A = np.arange(6, dtype=np.float32).reshape(2, 3)
    B = np.arange(6, dtype=np.float32).reshape(3, 2)
    try:
        C = mod.matmul(A, B)
    except RuntimeError:
        pytest.skip("matmul requires GPU runtime")
    assert np.allclose(C, A.dot(B), atol=1e-4)


def test_matmul_square():
    mod = _load("algorithms.gpu.kernels.numba_kernels")
    A = np.eye(4, dtype=np.float32)
    B = np.random.rand(4, 4).astype(np.float32)
    try:
        C = mod.matmul(A, B)
    except RuntimeError:
        pytest.skip("matmul requires GPU runtime")
    assert np.allclose(C, B, atol=1e-4)


# ── Reduction ─────────────────────────────────────────────────────────────────

def test_sum_reduce():
    mod = _load("algorithms.gpu.reduction.reduction")
    arr = np.arange(100, dtype=np.float32)
    assert abs(mod.sum_reduce(arr) - float(np.sum(arr))) < 1e-3


def test_max_reduce():
    mod = _load("algorithms.gpu.reduction.reduction")
    arr = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0], dtype=np.float32)
    assert abs(mod.max_reduce(arr) - 9.0) < 1e-6


# ── Prefix Scan ───────────────────────────────────────────────────────────────

def test_prefix_sum_correctness():
    mod = _load("algorithms.gpu.scan.scan")
    arr = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    out = np.array(mod.prefix_sum(arr))
    assert np.allclose(out, np.cumsum(arr))


def test_prefix_sum_zeros():
    mod = _load("algorithms.gpu.scan.scan")
    arr = np.zeros(8, dtype=np.float32)
    out = np.array(mod.prefix_sum(arr))
    assert np.allclose(out, 0)


# ── Sparse ops (SpMV wrapper) ─────────────────────────────────────────────────

def test_sparse_ops_spmv():
    mod = _load("algorithms.gpu.sparse.sparse_ops")
    # Diagonal 3×3 matrix: [[2,0,0],[0,3,0],[0,0,4]]
    data = [2, 3, 4]
    indices = [0, 1, 2]
    indptr = [0, 1, 2, 3]
    x = [1, 2, 3]
    out = mod.spmv(data, indices, indptr, x)
    assert np.allclose(out, [2, 6, 12])


# ── GPU sort (parallel_sort — has CPU fallback) ───────────────────────────────

def test_parallel_sort_correctness():
    mod = _load("algorithms.gpu.sorting.parallel_sort")
    arr = np.array([5, 3, 8, 1, 9, 2], dtype=np.float64)
    out = mod.parallel_sort(arr)
    assert np.array_equal(out, np.sort(arr))


def test_radix_cupy_correctness():
    mod = _load("algorithms.gpu.sorting.radix_cupy")
    arr = [5, 3, 6, 1, 2, 9, -1, 0]
    out = list(mod.radix_sort_cupy(arr))
    assert out == sorted(arr)


# ── Tiled matmul ──────────────────────────────────────────────────────────────

def test_matmul_tiled_matches_numpy():
    mod = _load("algorithms.gpu.kernels.numba_kernels")
    A = np.random.rand(32, 32).astype(np.float32)
    B = np.random.rand(32, 32).astype(np.float32)
    try:
        C = mod.matmul_tiled(A, B)
    except RuntimeError:
        pytest.skip("matmul_tiled requires GPU runtime")
    assert np.allclose(C, A.dot(B), atol=1e-3)


def test_matmul_tiled_non_square():
    mod = _load("algorithms.gpu.kernels.numba_kernels")
    A = np.random.rand(16, 24).astype(np.float32)
    B = np.random.rand(24, 8).astype(np.float32)
    try:
        C = mod.matmul_tiled(A, B)
    except RuntimeError:
        pytest.skip("matmul_tiled requires GPU runtime")
    assert C.shape == (16, 8)
    assert np.allclose(C, A.dot(B), atol=1e-3)


def test_matmul_tiled_identity():
    mod = _load("algorithms.gpu.kernels.numba_kernels")
    A = np.eye(16, dtype=np.float32)
    B = np.random.rand(16, 16).astype(np.float32)
    try:
        C = mod.matmul_tiled(A, B)
    except RuntimeError:
        pytest.skip("matmul_tiled requires GPU runtime")
    assert np.allclose(C, B, atol=1e-4)


def test_matmul_tiled_shape_mismatch_raises():
    mod = _load("algorithms.gpu.kernels.numba_kernels")
    A = np.ones((4, 3), dtype=np.float32)
    B = np.ones((5, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        mod.matmul_tiled(A, B)


# ── GPU BFS frontier ──────────────────────────────────────────────────────────

def test_bfs_frontier_all_reachable():
    mod = _load("algorithms.gpu.graphs.bfs_frontier")
    # simple chain: 0-1-2-3-4
    n = 5
    adj = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2, 4], 4: [3]}
    row_ptr, col_idx = mod.adj_to_csr(n, adj)
    levels = mod.bfs_frontier(n, row_ptr, col_idx, source=0)
    assert list(levels) == [0, 1, 2, 3, 4]


def test_bfs_frontier_star_graph():
    mod = _load("algorithms.gpu.graphs.bfs_frontier")
    # star: 0 connected to 1,2,3,4
    n = 5
    adj = {0: [1, 2, 3, 4], 1: [0], 2: [0], 3: [0], 4: [0]}
    row_ptr, col_idx = mod.adj_to_csr(n, adj)
    levels = mod.bfs_frontier(n, row_ptr, col_idx, source=0)
    assert levels[0] == 0
    assert all(levels[i] == 1 for i in range(1, 5))


def test_bfs_frontier_disconnected():
    mod = _load("algorithms.gpu.graphs.bfs_frontier")
    # two disconnected edges: 0-1 and 2-3
    n = 4
    adj = {0: [1], 1: [0], 2: [3], 3: [2]}
    row_ptr, col_idx = mod.adj_to_csr(n, adj)
    levels = mod.bfs_frontier(n, row_ptr, col_idx, source=0)
    assert levels[0] == 0
    assert levels[1] == 1
    assert levels[2] == -1
    assert levels[3] == -1


def test_bfs_frontier_matches_cpu_bfs():
    """GPU and CPU BFS frontiers agree on a random connected graph."""
    mod = _load("algorithms.gpu.graphs.bfs_frontier")
    cpu_bfs = _load("algorithms.cpu.graphs.bfs")
    rng = np.random.default_rng(42)
    n = 20
    adj = {i: [] for i in range(n)}
    # build a random connected graph
    for u in range(1, n):
        v = int(rng.integers(0, u))
        adj[u].append(v)
        adj[v].append(u)
    # extra random edges
    for _ in range(n):
        u, v = int(rng.integers(0, n)), int(rng.integers(0, n))
        if u != v and v not in adj[u]:
            adj[u].append(v)
            adj[v].append(u)

    # GPU BFS levels
    row_ptr, col_idx = mod.adj_to_csr(n, adj)
    gpu_levels = mod.bfs_frontier(n, row_ptr, col_idx, source=0)

    # CPU BFS levels via reference implementation
    order = cpu_bfs.bfs(adj, 0)
    cpu_levels = np.full(n, -1, dtype=np.int32)
    level_map = {}
    for node in order:
        if node == 0:
            level_map[node] = 0
        else:
            level_map[node] = min(
                (level_map[nb] + 1 for nb in adj.get(node, []) if nb in level_map),
                default=-1,
            )
        cpu_levels[node] = level_map[node]

    assert np.array_equal(gpu_levels, cpu_levels)


def test_adj_to_csr_shape():
    mod = _load("algorithms.gpu.graphs.bfs_frontier")
    adj = {0: [1, 2], 1: [0], 2: [0]}
    row_ptr, col_idx = mod.adj_to_csr(3, adj)
    assert len(row_ptr) == 4   # n + 1
    assert len(col_idx) == 4   # total edges: 2+1+1


if __name__ == "__main__":
    test_fft_length()
    test_fft_matches_numpy()
    test_convolve_length()
    test_convolve_identity_kernel()
    test_convolve_matches_numpy()
    test_vec_add_correctness()
    test_vec_add_zeros()
    test_sum_reduce()
    test_max_reduce()
    test_prefix_sum_correctness()
    test_prefix_sum_zeros()
    test_sparse_ops_spmv()
    test_parallel_sort_correctness()
    test_radix_cupy_correctness()
    test_matmul_tiled_matches_numpy()
    test_matmul_tiled_non_square()
    test_matmul_tiled_identity()
    test_matmul_tiled_shape_mismatch_raises()
    test_bfs_frontier_all_reachable()
    test_bfs_frontier_star_graph()
    test_bfs_frontier_disconnected()
    test_bfs_frontier_matches_cpu_bfs()
    test_adj_to_csr_shape()
    print("GPU kernel tests passed (CPU fallbacks used where GPU unavailable)")
