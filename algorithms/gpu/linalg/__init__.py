"""GPU linear algebra (norm, outer product, matrix multiply)."""
from algorithms.gpu.linalg.linalg import norm, outer_product
from algorithms.gpu.linalg.matrix_ops import matmul

__all__ = ["norm", "outer_product", "matmul"]
