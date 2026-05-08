"""GPU algorithm implementations.

Sub-packages mirror the CPU layout:

    gpu/
    ├── signal/       fft, convolution
    ├── reduction/    sum_reduce, max_reduce, min_reduce, mean_reduce, dot_product
    ├── scan/         prefix_sum, diff_gpu
    ├── elementwise/  argsort_gpu, clip_gpu, histogram_gpu, softmax_gpu
    ├── linalg/       norm, outer_product, matmul (CuPy)
    ├── sorting/      cupy_sort, parallel_sort, radix_sort_cupy
    ├── sparse/       spmv (CuPy CSR + Numba CUDA spMV)
    └── kernels/      vec_add, matmul (Numba CUDA with CPU fallback)

All sub-packages include CPU fallbacks so they are importable without a GPU.
"""
