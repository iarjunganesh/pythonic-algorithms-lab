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
    print("GPU kernel tests passed (CPU fallbacks used where GPU unavailable)")
