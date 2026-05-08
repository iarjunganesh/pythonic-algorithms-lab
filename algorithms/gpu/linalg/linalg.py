try:
    import cupy as cp
except Exception:
    cp = None


def norm(arr):
    """L2 (Euclidean) norm of a vector on GPU."""
    if cp is None:
        import numpy as np
        return float(np.linalg.norm(np.array(arr, dtype=np.float32)))
    a = cp.asarray(arr, dtype=cp.float32)
    return float(cp.asnumpy(cp.linalg.norm(a)))


def outer_product(arr):
    """Outer product of two equal-length halves of arr on GPU (O(n²/4) work)."""
    if cp is None:
        import numpy as np
        a = np.array(arr, dtype=np.float32)
        mid = max(1, len(a) // 2)
        return np.outer(a[:mid], a[mid:mid + mid])
    a = cp.asarray(arr, dtype=cp.float32)
    mid = max(1, len(a) // 2)
    return cp.asnumpy(cp.outer(a[:mid], a[mid:mid + mid]))
