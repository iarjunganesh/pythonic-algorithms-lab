"""GPU smoke test and environment detection for CI/local checks.

This script performs minimal runtime checks for CuPy and Numba CUDA kernels
and exercises a couple of small kernels with CPU fallbacks so it is safe to
run on machines without GPUs.
"""
import sys


def check_cupy():
    try:
        import cupy as cp
    except Exception as e:
        print('CuPy not available:', e)
        return False
    try:
        print('CuPy version:', cp.__version__)
        cnt = cp.cuda.runtime.getDeviceCount()
        print('CuPy device count:', cnt)
        a = cp.arange(16)
        s = int(cp.sum(a))
        print('CuPy sum test OK, sum=', s)
        return cnt > 0
    except Exception as e:
        print('CuPy runtime error:', e)
        return False


def check_numba():
    try:
        from algorithms.gpu.kernels.numba_kernels import vec_add, matmul
    except Exception as e:
        print('Numba kernels module unavailable or import failed:', e)
        return False
    try:
        # small CPU-fallback tests; functions include GPU fallbacks when available
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        out = vec_add(a, b)
        print('Numba vec_add OK, len=', len(out))
        A = [[1.0, 2.0], [3.0, 4.0]]
        B = [[1.0, 0.0], [0.0, 1.0]]
        C = matmul(A, B)
        print('Numba matmul OK, shape=', (len(C), len(C[0]) if C else 0))
        return True
    except Exception as e:
        print('Numba kernel runtime error:', e)
        return False


def main():
    cupy_ok = check_cupy()
    numba_ok = check_numba()
    ok = cupy_ok or numba_ok
    if ok:
        print('GPU smoke: at least one GPU path validated (or CPU fallbacks OK).')
        return 0
    print('GPU smoke failed: neither CuPy nor Numba kernels usable.')
    return 1


if __name__ == '__main__':
    sys.exit(main())
