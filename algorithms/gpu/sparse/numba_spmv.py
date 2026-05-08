"""Numba CUDA CSR sparse matrix-vector multiply (spMV) with CPU fallback."""
import numpy as np

try:
    from numba import cuda
    numba_cuda_available = cuda.is_available()
except Exception:
    cuda = None
    numba_cuda_available = False


if numba_cuda_available:
    @cuda.jit
    def _spmv_csr_kernel(data, indices, indptr, x, out):
        row = cuda.grid(1)
        if row < indptr.size - 1:
            start = indptr[row]
            end = indptr[row + 1]
            tmp = 0.0
            for j in range(start, end):
                tmp += data[j] * x[indices[j]]
            out[row] = tmp


def spmv_csr(data, indices, indptr, x):
    """Compute y = A @ x where A is CSR encoded by (data, indices, indptr).

    Uses Numba CUDA kernel when available; otherwise falls back to CPU dense
    computation for convenience (not optimized for large matrices).
    """
    data = np.asarray(data)
    indices = np.asarray(indices, dtype=np.int32)
    indptr = np.asarray(indptr, dtype=np.int32)
    x = np.asarray(x)
    n = indptr.size - 1
    out = np.zeros(n, dtype=data.dtype)

    if numba_cuda_available:
        d_data = cuda.to_device(data)
        d_indices = cuda.to_device(indices)
        d_indptr = cuda.to_device(indptr)
        d_x = cuda.to_device(x)
        d_out = cuda.device_array_like(out)
        threads_per_block = 128
        blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
        _spmv_csr_kernel[blocks_per_grid, threads_per_block](d_data, d_indices, d_indptr, d_x, d_out)
        return d_out.copy_to_host()

    # CPU fallback: compute using CSR loops
    for i in range(n):
        s = 0.0
        for j in range(indptr[i], indptr[i + 1]):
            s += data[j] * x[indices[j]]
        out[i] = s
    return out
