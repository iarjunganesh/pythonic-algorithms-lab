"""GPU sparse-matrix operations (SpMV via CuPy and Numba CUDA)."""
from algorithms.gpu.sparse.sparse_ops import spmv

__all__ = ["spmv"]
