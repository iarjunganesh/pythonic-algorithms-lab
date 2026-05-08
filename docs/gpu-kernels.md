# GPU Kernels & Installation

This document explains the optional GPU kernels in this repository and how to install and verify GPU toolchains.

## Package structure

GPU implementations live under `algorithms/gpu/` and mirror the CPU sub-package layout:

```
algorithms/gpu/
├── signal/         fft.py, convolution.py
├── reduction/      reduction.py  (sum, max, min, mean, dot product)
├── scan/           scan.py       (prefix_sum, diff_gpu)
├── elementwise/    elementwise.py (argsort, clip, histogram, softmax)
├── linalg/         linalg.py, matrix_ops.py  (norm, outer_product, matmul)
├── sorting/        gpu_sort.py, parallel_sort.py, radix_cupy.py
├── sparse/         sparse_ops.py, numba_spmv.py  (SpMV — CuPy + Numba)
└── kernels/        numba_kernels.py  (vec_add, matmul — Numba CUDA)
```

Each sub-package exposes a clean `__init__.py` so you can import at any level:

```python
from algorithms.gpu.signal import fft, convolve
from algorithms.gpu.reduction import sum_reduce, dot_product
from algorithms.gpu.linalg import norm, matmul
```

## GPU kernels at a glance (n = 100,000 · RTX 5070)

| Sub-package | Algorithm | n=100k (ms) |
|---|---|---|
| `signal` | `fft` | 3.5 |
| `signal` | `convolve` | 12.7 |
| `reduction` | `sum_reduce` | 2.6 |
| `reduction` | `max_reduce` | 2.5 |
| `reduction` | `min_reduce` | 2.5 |
| `reduction` | `mean_reduce` | 2.6 |
| `reduction` | `dot_product` | 2.7 |
| `scan` | `prefix_sum` | 2.6 |
| `scan` | `diff_gpu` | 2.6 |
| `elementwise` | `argsort_gpu` | 2.7 |
| `elementwise` | `clip_gpu` | 2.7 |
| `elementwise` | `histogram_gpu` | 3.3 |
| `elementwise` | `softmax_gpu` | 2.6 |
| `linalg` | `norm` | 2.6 |
| `linalg` | `outer_product` | 0.3 (capped n=1000) |
| `linalg` | `matmul_cupy` | 3.0 |
| `sorting` | `cupy_sort` | 3.5 |
| `sorting` | `gpu_sort` | 3.0 |
| `sorting` | `radix_cupy` | 3.3 |

*Numba CUDA (`kernels/`) is gated on `numba_cuda_available` — unavailable on Windows WDDM; falls back to NumPy CPU.*

## Supported backends

- **CuPy**: high-level GPU array operations used by all sub-packages except `kernels/`.
- **Numba CUDA**: custom kernels in `algorithms/gpu/kernels/numba_kernels.py` and `algorithms/gpu/sparse/numba_spmv.py`.

## Installation (summary)

1. Install a CuPy wheel that matches your CUDA runtime. For CUDA 13.x (example):

```bash
pip install cupy-cuda13x
```

2. Install Numba for CUDA kernels:

```bash
pip install numba
```

3. Optional: install dashboard tooling for interactive plots:

```bash
pip install plotly dash
```

## System prerequisites

- Ensure NVIDIA drivers and the CUDA runtime compatible with your CuPy wheel are installed on the host. On many systems `nvidia-smi` is a quick check for driver availability.
- Match the CuPy wheel to your installed CUDA runtime (mismatched CuPy/CUDA typically fails at import time).
- **Windows note**: Numba CUDA JIT requires TCC driver mode. Under the default WDDM mode (gaming/display driver) `cuda.is_available()` returns `False` and all Numba kernels silently fall back to NumPy CPU.

## Verifying your GPU environment

In Python, run this snippet to verify CuPy and Numba availability:

```python
import importlib
import sys

print('Python:', sys.version.splitlines()[0])
print('cupy present:', importlib.util.find_spec('cupy') is not None)
try:
    import cupy as cp
    print('cupy version:', cp.__version__)
    print('cupy can access device count:', cp.cuda.runtime.getDeviceCount())
except Exception as e:
    print('cupy import error:', e)

try:
    import numba
    print('numba version:', numba.__version__)
    print('numba.cuda.is_available():', getattr(numba.cuda, 'is_available', lambda: False)())
except Exception as e:
    print('numba import error:', e)
```

Quick runtime test (CuPy):

```python
import cupy as cp
arr = cp.arange(10)
print('sum:', int(cp.sum(arr)))
```

## Testing and smoke tests

- Use `pytest tests/test_gpu_kernels.py` to run the GPU kernel tests. All tests use CPU fallbacks if CuPy/Numba are absent, making them safe on CI runners without GPUs.
- The repository includes an optional GitHub Actions job that can run GPU smoke tests on a self-hosted runner labeled `gpu`. See `.github/workflows/ci.yml` for details.

## Running the canonical benchmark

```bash
# small smoke run (fast):
python benchmarks/run_benchmarks.py --full --sizes 10 100 --repeat 2 --out benchmarks/results_smoke.csv

# canonical full sweep:
python benchmarks/run_benchmarks.py --full --sizes 100 1000 5000 10000 50000 100000 --repeat 5 --out benchmarks/results_full.csv
```

To verify GPU runtime specifically:

```bash
python benchmarks/gpu_smoke.py
```

The runner only registers GPU-backed implementations when a compatible GPU toolchain is detected; otherwise CPU fallbacks are used.

## Conda-based install (recommended for Numba/CUDA)

```bash
conda create -n pyalg python=3.14 -c conda-forge numba cudatoolkit
conda activate pyalg
pip install -r requirements.txt
```

## Notes and best practices

- Prefer CuPy primitives for most array-heavy workloads — they are highly optimized and use vendor-tuned algorithms.
- Custom Numba kernels are useful for algorithmic exploration (e.g., SpMV, custom scans) and for pedagogical comparisons; they may not always outperform tuned libraries.
- Keep CPU fallbacks in tests and examples so contributors without GPUs can run and validate code.
- GPU kernel launch overhead (~0.15 ms on RTX 5070) means CPU beats GPU below n ≈ 1,000–10,000 depending on the operation.
