# Algorithm Complexity Reference

Big-O time and space complexities for every algorithm implemented in this repository.

## Sorting

| Algorithm | Best | Average | Worst | Space |
|---|---|---|---|---|
| Bubble Sort | O(n) | O(n²) | O(n²) | O(1) |
| Insertion Sort | O(n) | O(n²) | O(n²) | O(1) |
| Selection Sort | O(n²) | O(n²) | O(n²) | O(1) |
| Merge Sort | O(n log n) | O(n log n) | O(n log n) | O(n) |
| Quick Sort | O(n log n) | O(n log n) | O(n²) | O(log n) avg |
| Heap Sort | O(n log n) | O(n log n) | O(n log n) | O(1) |
| Shell Sort | O(n log n) | gap-dependent | O(n²) | O(1) |
| Counting Sort | O(n + k) | O(n + k) | O(n + k) | O(k) |
| Radix Sort | O(d(n + k)) | O(d(n + k)) | O(d(n + k)) | O(n + k) |
| Timsort | O(n) | O(n log n) | O(n log n) | O(n) |

*k = range of values, d = number of digits*

## Searching

| Algorithm | Best | Average | Worst | Space | Notes |
|---|---|---|---|---|---|
| Linear Search | O(1) | O(n) | O(n) | O(1) | |
| Binary Search | O(1) | O(log n) | O(log n) | O(1) | Requires sorted input |
| Jump Search | O(1) | O(√n) | O(√n) | O(1) | Requires sorted input |
| Fibonacci Search | O(1) | O(log n) | O(log n) | O(1) | Requires sorted input |

## Graphs

| Algorithm | Time | Space | Notes |
|---|---|---|---|
| BFS | O(V + E) | O(V) | |
| DFS | O(V + E) | O(V) | |
| Dijkstra | O(E + V log V) | O(V) | Binary heap; non-negative weights only |

*V = vertices, E = edges*

## Dynamic Programming

| Algorithm | Time | Space |
|---|---|---|
| Fibonacci (memoized) | O(n) | O(n) |
| 0/1 Knapsack | O(nW) | O(nW) |

*n = number of items, W = capacity*

## Data Structures

| Structure | Insert | Search | Delete | Space | Notes |
|---|---|---|---|---|---|
| Stack | O(1) | O(n) | O(1) | O(n) | LIFO |
| Queue | O(1) | O(n) | O(1) | O(n) | FIFO |
| Linked List | O(1) head | O(n) | O(n) | O(n) | |
| Doubly Linked List | O(1) head/tail | O(n) | O(1) with ref | O(n) | |
| BST | O(log n) avg | O(log n) avg | O(log n) avg | O(n) | Degrades to O(n) unbalanced |
| AVL Tree | O(log n) | O(log n) | O(log n) | O(n) | Self-balancing BST |
| Trie | O(k) | O(k) | O(k) | O(alphabet × n) | k = key length |
| SkipList | O(log n) avg | O(log n) avg | O(log n) avg | O(n log n) | Probabilistic |
| Bloom Filter | O(k) | O(k) | N/A | O(m) | Probabilistic; no false negatives |
| MinHeap | O(log n) push | O(1) peek | O(log n) pop | O(n) | |
| UnionFind | O(α(n)) | O(α(n)) | N/A | O(n) | α = inverse Ackermann (≈ O(1) practical) |

*k = hash functions (BloomFilter) or key length (Trie)*

## Strings

| Algorithm | Time | Space | Notes |
|---|---|---|---|
| KMP Search | O(n + m) | O(m) | n = text length, m = pattern length |
| Rabin-Karp | O(n + m) avg, O(nm) worst | O(1) | Rolling hash |
| Suffix Array | O(n log n) | O(n) | |

## Geometry

| Algorithm | Time | Space |
|---|---|---|
| Convex Hull (monotone chain) | O(n log n) | O(n) |
| Closest Pair (brute force) | O(n²) | O(1) |

## Math

| Algorithm | Time | Space |
|---|---|---|
| GCD (Euclidean) | O(log min(a, b)) | O(1) |
| Sieve of Eratosthenes | O(n log log n) | O(n) |

## Backtracking

| Algorithm | Time | Space | Notes |
|---|---|---|---|
| N-Queens | O(n!) worst | O(n) | Pruning reduces practical runtime significantly |

## Greedy

| Algorithm | Time | Space | Notes |
|---|---|---|---|
| Coin Change (greedy) | O(n log n) | O(n) | Optimal only for canonical coin systems |

## GPU Kernels

All GPU kernels fall back to NumPy CPU when CuPy/Numba CUDA is unavailable.

### Signal (`algorithms/gpu/signal/`)

| Algorithm | Time (GPU) | Notes |
|---|---|---|
| FFT | O(n log n) | `cp.fft.fft` |
| Convolution | O(n + m) | `cp.convolve` |

### Reduction (`algorithms/gpu/reduction/`)

| Algorithm | Time (GPU) | Notes |
|---|---|---|
| sum_reduce | O(n) | `cp.sum` |
| max_reduce | O(n) | `cp.max` |
| min_reduce | O(n) | `cp.min` |
| mean_reduce | O(n) | `cp.mean` |
| dot_product | O(n) | `cp.dot(a, a)` — self dot (sum of squares) |

### Scan (`algorithms/gpu/scan/`)

| Algorithm | Time (GPU) | Notes |
|---|---|---|
| prefix_sum | O(n) | `cp.cumsum` — inclusive scan |
| diff_gpu | O(n) | `cp.diff` — first-order finite differences |

### Elementwise (`algorithms/gpu/elementwise/`)

| Algorithm | Time (GPU) | Notes |
|---|---|---|
| argsort_gpu | O(n log n) | `cp.argsort` |
| clip_gpu | O(n) | `cp.clip` |
| histogram_gpu | O(n) | `cp.histogram` |
| softmax_gpu | O(n) | exp-normalise — core neural-net primitive |

### Linear Algebra (`algorithms/gpu/linalg/`)

| Algorithm | Time (GPU) | Notes |
|---|---|---|
| norm | O(n) | `cp.linalg.norm` — L2 Euclidean norm |
| outer_product | O(n²) | `cp.outer` — benchmarked capped at n=1 000 |
| matmul (CuPy) | O(n³) / optimised | `cp.matmul` via cuBLAS |

### Sorting (`algorithms/gpu/sorting/`)

| Algorithm | Time (GPU) | Notes |
|---|---|---|
| cupy_sort | O(n log n) | `cp.sort` — raises if CuPy unavailable |
| gpu_sort (parallel) | O(n log n) | `cp.sort` with CPU fallback |
| radix_cupy | O(n log n) | wraps `cp.sort`; extensible for custom radix |

### Sparse (`algorithms/gpu/sparse/`)

| Algorithm | Time (GPU) | Notes |
|---|---|---|
| SpMV (spmv) | O(nnz) | CuPy CSR or Numba CUDA kernel; nnz = non-zeros |

### Kernels (`algorithms/gpu/kernels/`)

| Algorithm | Time (GPU) | Notes |
|---|---|---|
| vec_add | O(n) | Numba CUDA; CPU fallback on WDDM Windows |
| matmul (Numba) | O(n³) | Numba CUDA; CPU fallback on WDDM Windows |


## GPU Kernels

All GPU kernels include CPU fallbacks. Wall-clock time is often dominated by host ↔ device transfer for small inputs; prefer large n when comparing CPU vs GPU.

| Kernel | Module | Backend | Notes |
|---|---|---|---|
| FFT | `gpu/fft.py` | CuPy / NumPy | `fft(arr)` |
| 1D Convolution | `gpu/convolution.py` | CuPy / NumPy | `convolve(a, b)` |
| Vec Add | `gpu/numba_kernels.py` | Numba CUDA / NumPy | `vec_add(a, b)` |
| Matmul (Numba) | `gpu/numba_kernels.py` | Numba CUDA / NumPy | `matmul(A, B)` |
| Matmul (CuPy) | `gpu/math/gpu_matrix_ops.py` | CuPy only | raises if no GPU |
| Sum Reduce | `gpu/reduction.py` | CuPy / NumPy | `sum_reduce(arr)` |
| Max Reduce | `gpu/reduction.py` | CuPy / NumPy | `max_reduce(arr)` |
| Prefix Sum | `gpu/scan.py` | CuPy / NumPy | `prefix_sum(arr)` |
| SpMV (Numba) | `gpu/numba_spmv.py` | Numba CUDA / CPU loops | `spmv_csr(data, indices, indptr, x)` |
| SpMV (wrapper) | `gpu/sparse_ops.py` | Numba → CuPy → CPU | `spmv(data, indices, indptr, x)` |
| GPU Sort | `gpu/sorting/gpu_sort.py` | CuPy only | `cupy_sort(arr)`, raises if no GPU |
| Parallel Sort | `gpu/sorting/parallel_sort.py` | CuPy / NumPy | `parallel_sort(arr)` |
| Radix Sort (CuPy) | `gpu/radix_cupy.py` | CuPy / NumPy | `radix_sort_cupy(arr)` |

## Benchmarking notes

- Include data-transfer overhead in GPU timings for honest CPU vs GPU comparisons. For small n the transfer cost usually dominates.
- Use `benchmarks/run_benchmarks.py` to generate reproducible CSV outputs for analysis.
- Big-O expresses asymptotic *work*; GPUs reduce wall-clock time by executing many operations in parallel. An O(n log n) sort is still O(n log n) on GPU but with a much smaller constant for large n.
- Graph traversals (BFS/DFS) have irregular memory patterns and typically benefit less from GPU parallelism than dense linear algebra or reduction primitives.
