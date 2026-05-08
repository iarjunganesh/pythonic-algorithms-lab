try:
    import cupy as cp
except Exception:
    cp = None

def parallel_sort(arr):
    """Sort using GPU (CuPy) if available, otherwise fall back to NumPy."""
    if cp is None:
        import numpy as np

        return np.sort(arr)
    a = cp.asarray(arr)
    s = cp.sort(a)
    return cp.asnumpy(s)
