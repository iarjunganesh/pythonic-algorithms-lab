try:
    import cupy as cp
except Exception:
    cp = None


def cupy_sort(arr):
    """Sort using CuPy on GPU and return a NumPy array on CPU.

    Raises RuntimeError if CuPy is not available.
    """
    if cp is None:
        raise RuntimeError("CuPy is not available")
    a = cp.asarray(arr)
    sorted_gpu = cp.sort(a)
    return cp.asnumpy(sorted_gpu)
