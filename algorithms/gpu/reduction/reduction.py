try:
    import cupy as cp
except Exception:
    cp = None

def sum_reduce(arr):
    """Compute sum using GPU if available, otherwise use NumPy."""
    if cp is None:
        import numpy as np
        return float(np.sum(arr))
    return float(cp.asnumpy(cp.sum(cp.asarray(arr))))

def max_reduce(arr):
    if cp is None:
        import numpy as np
        return float(np.max(arr))
    return float(cp.asnumpy(cp.max(cp.asarray(arr))))

def min_reduce(arr):
    if cp is None:
        import numpy as np
        return float(np.min(arr))
    return float(cp.asnumpy(cp.min(cp.asarray(arr))))

def mean_reduce(arr):
    if cp is None:
        import numpy as np
        return float(np.mean(arr))
    return float(cp.asnumpy(cp.mean(cp.asarray(arr, dtype=cp.float32))))

def dot_product(arr):
    """Self dot-product: sum of squares — classic GPU reduction primitive."""
    if cp is None:
        import numpy as np
        a = np.array(arr, dtype=np.float32)
        return float(np.dot(a, a))
    a = cp.asarray(arr, dtype=cp.float32)
    return float(cp.asnumpy(cp.dot(a, a)))
