try:
    import cupy as cp
except Exception:
    cp = None


def radix_sort_cupy(arr):
    """GPU-accelerated radix/sort using CuPy when available; falls back to NumPy.

    This wrapper uses CuPy's GPU sort (which is highly optimized). For integer-only
    workloads where a custom radix approach is needed, this can be extended.
    """
    if cp is None:
        import numpy as np

        return list(np.sort(arr))
    a = cp.asarray(arr)
    s = cp.sort(a)
    return cp.asnumpy(s)
