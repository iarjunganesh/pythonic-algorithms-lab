try:
    import cupy as cp
    from cupyx.scipy.sparse import csr_matrix
except Exception:
    cp = None
    csr_matrix = None

try:
    from algorithms.gpu.sparse.numba_spmv import spmv_csr as numba_spmv
except Exception:
    numba_spmv = None


def spmv(data, indices, indptr, x):
    """Sparse matrix-vector multiply using CSR representation on GPU when possible.

    Priority: Numba CUDA kernel (if available) -> CuPy CSR (if available) -> CPU fallback.
    """
    # Try Numba CUDA spMV first
    if numba_spmv is not None:
        try:
            return numba_spmv(data, indices, indptr, x)
        except Exception:
            pass

    # Next try CuPy CSR
    if csr_matrix is not None and cp is not None:
        A = csr_matrix((cp.asarray(data), cp.asarray(indices), cp.asarray(indptr)))
        return cp.asnumpy(A.dot(cp.asarray(x)))

    # CPU fallback
    import numpy as np

    n = len(indptr) - 1
    out = np.zeros(n, dtype=np.array(data).dtype)
    for i in range(n):
        s = 0.0
        for j in range(indptr[i], indptr[i + 1]):
            s += data[j] * x[indices[j]]
        out[i] = s
    return out
