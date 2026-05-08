try:
    import cupy as cp
except Exception:
    cp = None

def convolve(a, b):
    """1D convolution using GPU (CuPy) when available, otherwise NumPy."""
    if cp is None:
        import numpy as np

        return np.convolve(a, b)
    return cp.asnumpy(cp.convolve(cp.asarray(a), cp.asarray(b)))
