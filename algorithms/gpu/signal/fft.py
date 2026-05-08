try:
    import cupy as cp
except Exception:
    cp = None

def fft(arr):
    if cp is None:
        import numpy as np

        return np.fft.fft(arr)
    return cp.asnumpy(cp.fft.fft(cp.asarray(arr)))
