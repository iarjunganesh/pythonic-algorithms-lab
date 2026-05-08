"""Example Numba CUDA kernels with safe CPU fallbacks.

This module provides small GPU kernels (vector add, matrix multiply) implemented
with Numba CUDA. If Numba/CUDA is unavailable, functions fall back to NumPy.
"""
import numpy as np

try:
    from numba import cuda
    numba_cuda_available = cuda.is_available()
except Exception:
    cuda = None
    numba_cuda_available = False


if numba_cuda_available:
    @cuda.jit
    def _vec_add_kernel(a, b, out):
        i = cuda.grid(1)
        if i < out.size:
            out[i] = a[i] + b[i]

    @cuda.jit
    def _matmul_kernel(A, B, C):
        i, j = cuda.grid(2)
        if i < C.shape[0] and j < C.shape[1]:
            tmp = 0.0
            for k in range(A.shape[1]):
                tmp += A[i, k] * B[k, j]
            C[i, j] = tmp


def vec_add(a, b):
    """Element-wise add two arrays (GPU via Numba when available)."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError("shapes must match")
    if numba_cuda_available:
        d_a = cuda.to_device(a)
        d_b = cuda.to_device(b)
        d_out = cuda.device_array_like(d_a)
        threads_per_block = 256
        blocks_per_grid = (a.size + threads_per_block - 1) // threads_per_block
        _vec_add_kernel[blocks_per_grid, threads_per_block](d_a, d_b, d_out)
        return d_out.copy_to_host()
    else:
        return a + b


def matmul(A, B):
    """Matrix multiply A @ B (GPU via Numba when available)."""
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("only 2D matrices supported")
    if A.shape[1] != B.shape[0]:
        raise ValueError("shapes not aligned for matmul")
    if numba_cuda_available:
        d_A = cuda.to_device(A)
        d_B = cuda.to_device(B)
        d_C = cuda.device_array((A.shape[0], B.shape[1]), dtype=np.float32)
        threads_per_block = (16, 16)
        blocks_per_grid_x = (A.shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
        blocks_per_grid_y = (B.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
        _matmul_kernel[(blocks_per_grid_x, blocks_per_grid_y), threads_per_block](d_A, d_B, d_C)
        return d_C.copy_to_host()
    else:
        return A.dot(B)
