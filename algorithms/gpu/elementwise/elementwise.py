try:
    import cupy as cp
except Exception:
    cp = None


def argsort_gpu(arr):
    """Return indices that would sort the array, computed on GPU."""
    if cp is None:
        import numpy as np
        return np.argsort(arr)
    return cp.asnumpy(cp.argsort(cp.asarray(arr)))


def clip_gpu(arr, lo=0, hi=None):
    """Clip array values to [lo, hi] on GPU."""
    if cp is None:
        import numpy as np
        a = np.array(arr)
        return np.clip(a, lo, hi if hi is not None else a.max())
    a = cp.asarray(arr, dtype=cp.float32)
    hi_val = float(cp.max(a)) if hi is None else hi
    return cp.asnumpy(cp.clip(a, lo, hi_val))


def histogram_gpu(arr, bins=256):
    """Compute histogram on GPU."""
    if cp is None:
        import numpy as np
        return np.histogram(arr, bins=bins)
    a = cp.asarray(arr, dtype=cp.float32)
    return cp.asnumpy(cp.histogram(a, bins=bins)[0])


def softmax_gpu(arr):
    """Compute softmax on GPU — core neural network primitive."""
    if cp is None:
        import numpy as np
        a = np.array(arr, dtype=np.float32)
        e = np.exp(a - np.max(a))
        return e / e.sum()
    a = cp.asarray(arr, dtype=cp.float32)
    e = cp.exp(a - cp.max(a))
    return cp.asnumpy(e / e.sum())
