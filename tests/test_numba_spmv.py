import numpy as np

from algorithms.gpu.sparse.numba_spmv import spmv_csr


def test_spmv_small():
    # 3x3 matrix:
    # [10 0 0]
    # [0 20 0]
    # [1 2 3]
    data = [10, 20, 1, 2, 3]
    indices = [0, 1, 0, 1, 2]
    indptr = [0, 1, 2, 5]
    x = [1, 1, 1]
    y = spmv_csr(data, indices, indptr, x)
    assert np.allclose(y, [10, 20, 6])


if __name__ == "__main__":
    test_spmv_small()
    print("Numba spMV test passed (or CPU fallback was used)")
