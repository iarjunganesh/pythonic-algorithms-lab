try:
    import cupy as cp
except Exception:
    cp = None

def prefix_sum(arr):
    """Prefix-sum (exclusive or inclusive) using GPU if available."""
    if cp is None:
        import numpy as np
        return np.cumsum(arr)
    return cp.asnumpy(cp.cumsum(cp.asarray(arr)))

def diff_gpu(arr):
    """First-order finite differences on GPU (cp.diff)."""
    if cp is None:
        import numpy as np
        return np.diff(np.array(arr))
    return cp.asnumpy(cp.diff(cp.asarray(arr, dtype=cp.float32)))
