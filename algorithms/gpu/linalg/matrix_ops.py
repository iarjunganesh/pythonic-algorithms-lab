try:
    import cupy as cp
except Exception:
    cp = None


def matmul(a, b):
    """Multiply two matrices using CuPy; returns NumPy array on CPU.

    Raises RuntimeError if CuPy is unavailable.
    """
    if cp is None:
        raise RuntimeError("CuPy is not available")
    A = cp.asarray(a)
    B = cp.asarray(b)
    C = A @ B
    return cp.asnumpy(C)
