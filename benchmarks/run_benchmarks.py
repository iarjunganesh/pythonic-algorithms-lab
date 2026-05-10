"""Extensible benchmark runner with group, full-sweep, and merge modes.

Usage examples:
    python benchmarks/run_benchmarks.py --group sort --sizes 100 1000 --repeat 3 --out benchmarks/results.csv
    python benchmarks/run_benchmarks.py --full --sizes 1000 10000 50000 --repeat 3 --out benchmarks/results_full.csv
    python benchmarks/run_benchmarks.py --merge-inputs "benchmarks/results_*.csv" --merge-out benchmarks/results_combined.csv
"""
import argparse
import atexit
import csv
import glob
import os
import signal
from pathlib import Path
import random
import sys

# ---------------------------------------------------------------------------
# Single-instance lock — prevents accidental duplicate benchmark runs.
# CuPy/Numba spawn a *worker* child process on Windows; that child will NOT
# run __main__, so the lock is only acquired by the true top-level invocation.
# ---------------------------------------------------------------------------
_LOCK_PATH = Path(__file__).parent / ".run_benchmarks.lock"
_LOCK_FD: int | None = None


def _acquire_lock() -> bool:
    """Try to acquire the lock file. Returns False if already locked."""
    global _LOCK_FD
    try:
        _LOCK_FD = os.open(str(_LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_RDWR)
        _LOCK_PATH.write_text(str(os.getpid()))
        return True
    except FileExistsError:
        # Check if the PID in the lock is actually alive
        try:
            pid = int(_LOCK_PATH.read_text().strip())
            os.kill(pid, 0)   # signal 0 = check existence only
            return False       # process is alive — real duplicate
        except (ProcessLookupError, PermissionError, ValueError, OSError):
            # Stale lock from a crashed run — remove and acquire
            _LOCK_PATH.unlink(missing_ok=True)
            return _acquire_lock()


def _release_lock(*_):
    global _LOCK_FD
    if _LOCK_FD is not None:
        try:
            os.close(_LOCK_FD)
        except OSError:
            pass
        _LOCK_FD = None
    _LOCK_PATH.unlink(missing_ok=True)


atexit.register(_release_lock)
for _sig in (signal.SIGINT, signal.SIGTERM):
    try:
        signal.signal(_sig, _release_lock)
    except (OSError, ValueError):
        pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.data_gen import random_list
from core.decorators import benchmark

from algorithms.cpu.sorting.bubble_sort import bubble_sort
from algorithms.cpu.sorting.merge_sort import merge_sort
from algorithms.cpu.sorting.quick_sort import quick_sort
from algorithms.cpu.sorting.heap_sort import heap_sort
from algorithms.cpu.sorting.tim_sort import tim_sort
from algorithms.cpu.searching.binary_search import binary_search
from algorithms.cpu.searching.linear_search import linear_search
from algorithms.cpu.searching.jump_search import jump_search
from algorithms.cpu.searching.fibonacci_search import fibonacci_search
from algorithms.cpu.graphs.bfs import bfs
from algorithms.cpu.graphs.dijkstra import dijkstra
from algorithms.cpu.graphs.dfs import dfs

# additional CPU sorts
from algorithms.cpu.sorting.insertion_sort import insertion_sort
from algorithms.cpu.sorting.selection_sort import selection_sort
from algorithms.cpu.sorting.shell_sort import shell_sort
from algorithms.cpu.sorting.counting_sort import counting_sort
from algorithms.cpu.sorting.radix_sort import radix_sort

# additional CPU data-structures
from algorithms.cpu.data_structures.trie import Trie
from algorithms.cpu.data_structures.skiplist import SkipList
from algorithms.cpu.data_structures.bst import BST
from algorithms.cpu.data_structures.avl import insert as avl_insert
from algorithms.cpu.data_structures.bloom_filter import BloomFilter
from algorithms.cpu.data_structures.heap_wrapper import MinHeap
from algorithms.cpu.data_structures.doubly_linked_list import DoublyLinkedList

try:
    from algorithms.gpu.sorting.parallel_sort import parallel_sort
except Exception:
    parallel_sort = None
try:
    from algorithms.gpu.sorting.gpu_sort import cupy_sort
except Exception:
    cupy_sort = None

try:
    from algorithms.gpu.sorting.radix_cupy import radix_sort_cupy
except Exception:
    radix_sort_cupy = None

try:
    from algorithms.gpu.signal.fft import fft as gpu_fft
except Exception:
    gpu_fft = None

try:
    from algorithms.gpu.signal.convolution import convolve as gpu_convolve
except Exception:
    gpu_convolve = None

try:
    from algorithms.gpu.scan.scan import prefix_sum as gpu_prefix_sum, diff_gpu as gpu_diff
except Exception:
    gpu_prefix_sum = None
    gpu_diff = None

try:
    from algorithms.gpu.reduction.reduction import (
        sum_reduce as gpu_sum_reduce,
        max_reduce as gpu_max_reduce,
        min_reduce as gpu_min_reduce,
        mean_reduce as gpu_mean_reduce,
        dot_product as gpu_dot_product,
    )
except Exception:
    gpu_sum_reduce = None
    gpu_max_reduce = None
    gpu_min_reduce = None
    gpu_mean_reduce = None
    gpu_dot_product = None

try:
    from algorithms.gpu.kernels.numba_kernels import vec_add as gpu_vec_add, matmul as gpu_matmul_numba
    from algorithms.gpu.kernels.numba_kernels import matmul_tiled as gpu_matmul_tiled
    from algorithms.gpu.kernels.numba_kernels import numba_cuda_available as _numba_cuda_available
except Exception:
    gpu_vec_add = None
    gpu_matmul_numba = None
    gpu_matmul_tiled = None
    _numba_cuda_available = False

try:
    from algorithms.gpu.sparse.sparse_ops import spmv as gpu_spmv
except Exception:
    gpu_spmv = None

try:
    from algorithms.gpu.linalg.matrix_ops import matmul as gpu_matmul_cupy
except Exception:
    gpu_matmul_cupy = None

try:
    from algorithms.gpu.elementwise.elementwise import argsort_gpu, clip_gpu, histogram_gpu, softmax_gpu
except Exception:
    argsort_gpu = None
    clip_gpu = None
    histogram_gpu = None
    softmax_gpu = None

try:
    from algorithms.gpu.linalg.linalg import norm as gpu_norm, outer_product as gpu_outer_product
except Exception:
    gpu_norm = None
    gpu_outer_product = None

try:
    from algorithms.gpu.graphs.bfs_frontier import bfs_frontier, adj_to_csr as _adj_to_csr
except Exception:
    bfs_frontier = None
    _adj_to_csr = None

GROUPS = {
    "sort": {
        "cpu": {
            "bubble": bubble_sort,
            "merge": merge_sort,
            "quick": quick_sort,
            "heap": heap_sort,
            "timsort": tim_sort,
        },
        "gpu": {
            "gpu_sort": parallel_sort,
        },
    }
}


def _make_search_wrappers():
    def linear(data):
        if not data:
            return -1
        # use max+1 as target: guaranteed not in list, forces full worst-case scan
        return linear_search(data, max(data) + 1)

    def binary(data):
        if not data:
            return -1
        s = sorted(data)
        target = s[len(s) // 2]
        return binary_search(s, target)

    def jump(data):
        if not data:
            return -1
        s = sorted(data)
        target = s[len(s) // 2]
        return jump_search(s, target)

    def fib(data):
        if not data:
            return -1
        s = sorted(data)
        target = s[len(s) // 2]
        return fibonacci_search(s, target)

    return {"linear": linear, "binary": binary, "jump": jump, "fibonacci": fib}


def _make_graph_wrappers():
    def _rand_graph(n, avg_deg=4):
        adj = {i: [] for i in range(n)}
        for u in range(n):
            for _ in range(avg_deg):
                v = random.randrange(n)
                if v != u and v not in adj[u]:
                    adj[u].append(v)
        return adj

    def run_bfs(data):
        n = max(2, min(5000, len(data)))  # cap: large graphs are very slow in Python
        g = _rand_graph(n)
        return bfs(g, 0)

    def run_dijkstra(data):
        n = max(2, min(5000, len(data)))  # cap: large graphs are very slow in Python
        adj = {i: [] for i in range(n)}
        for u in range(n):
            for _ in range(3):
                v = random.randrange(n)
                if v != u:
                    adj[u].append((v, random.random() * 10 + 0.1))
        return dijkstra(adj, 0)

    def run_dfs(data):
        import sys
        n = max(2, min(1000, len(data)))  # cap to avoid recursion limit
        g = _rand_graph(n)
        old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(max(old_limit, n * 4 + 200))
        try:
            return dfs(g, 0)
        finally:
            sys.setrecursionlimit(old_limit)

    return {"bfs": run_bfs, "dijkstra": run_dijkstra, "dfs": run_dfs}


def _make_graph_gpu_wrappers():
    def _rand_csr(n, avg_deg=4):
        adj = {i: [] for i in range(n)}
        for u in range(n):
            for _ in range(avg_deg):
                v = random.randrange(n)
                if v != u and v not in adj[u]:
                    adj[u].append(v)
        return _adj_to_csr(n, adj)

    def run_bfs_frontier(data):
        n = max(2, min(5000, len(data)))
        row_ptr, col_idx = _rand_csr(n)
        return bfs_frontier(n, row_ptr, col_idx, source=0)

    return {"bfs_frontier": run_bfs_frontier}


GROUPS.update({
    "search": {"cpu": _make_search_wrappers(), "gpu": {}},
    "graphs": {
        "cpu": _make_graph_wrappers(),
        "gpu": _make_graph_gpu_wrappers() if bfs_frontier is not None else {},
    },
})

# Additional algorithm groups and wrappers
from algorithms.cpu.dynamic_prog.fib_memo import fib as fib_dp
from algorithms.cpu.dynamic_prog.knapsack import knapsack_01

from algorithms.cpu.strings.kmp import kmp_search
from algorithms.cpu.strings.rabin_karp import rabin_karp
from algorithms.cpu.strings.suffix_array import suffix_array

from algorithms.cpu.math.sieve import sieve
from algorithms.cpu.math.gcd import gcd

from algorithms.cpu.geometry.convex_hull import convex_hull
from algorithms.cpu.geometry.closest_pair import closest_pair

from algorithms.cpu.backtracking.n_queens import solve_n_queens

from algorithms.cpu.greedy.coin_change import coin_change_greedy

from algorithms.cpu.data_structures.stack import Stack
from algorithms.cpu.data_structures.queue import Queue
from algorithms.cpu.data_structures.linked_list import LinkedList
from algorithms.cpu.data_structures.union_find import UnionFind


def _make_dynamic_wrappers():
    def fib_wrap(data):
        n = max(5, min(30, max(1, len(data) // 100)))
        return fib_dp(n)

    def knap_wrap(data):
        m = max(5, min(40, max(1, len(data) // 25)))
        weights = [i % 10 + 1 for i in range(m)]
        values = [i % 20 + 1 for i in range(m)]
        W = sum(weights) // 2
        return knapsack_01(weights, values, W)

    return {"fib": fib_wrap, "knapsack": knap_wrap}


def _make_string_wrappers():
    def _make_text(n):
        # deterministic pseudo-random but fast text generator
        n = min(n, 20000)
        return ''.join(chr(97 + (i * 31) % 26) for i in range(n))

    def kmp_wrap(data):
        text = _make_text(min(max(100, len(data) * 10), 20000))
        pat = text[len(text) // 2: len(text) // 2 + max(1, min(10, len(text) // 10))]
        return kmp_search(text, pat)

    def rabin_wrap(data):
        text = _make_text(min(max(100, len(data) * 10), 20000))
        pat = text[len(text) // 3: len(text) // 3 + max(1, min(10, len(text) // 10))]
        return rabin_karp(text, pat)

    def suffix_wrap(data):
        s = _make_text(min(max(50, len(data) * 5), 2000))
        return suffix_array(s)

    return {"kmp": kmp_wrap, "rabin": rabin_wrap, "suffix_array": suffix_wrap}


def _make_math_wrappers():
    def sieve_wrap(data):
        n = max(100, len(data))
        return sieve(n)

    def gcd_wrap(data):
        a = sum(data) + 1
        b = sum(data[:len(data)//2] or [1]) + 2
        return gcd(a, b)

    # CPU numpy primitives — reference baselines for GPU comparison
    def fft_numpy(data):
        import numpy as _np
        return _np.fft.fft(_np.array(data, dtype=_np.float32))

    def prefix_sum_numpy(data):
        import numpy as _np
        return _np.cumsum(_np.array(data, dtype=_np.float32))

    def sum_reduce_numpy(data):
        import numpy as _np
        return float(_np.sum(_np.array(data, dtype=_np.float32)))

    def max_reduce_numpy(data):
        import numpy as _np
        return float(_np.max(_np.array(data, dtype=_np.float32)))

    def matmul_numpy(data):
        import numpy as _np
        side = max(2, int(len(data) ** 0.5))
        A = _np.array(data[:side * side], dtype=_np.float32).reshape(side, side)
        return A.dot(A)

    return {
        "sieve": sieve_wrap,
        "gcd": gcd_wrap,
        "fft": fft_numpy,
        "prefix_sum": prefix_sum_numpy,
        "sum_reduce": sum_reduce_numpy,
        "max_reduce": max_reduce_numpy,
        "matmul": matmul_numpy,
    }


def _make_geometry_wrappers():
    def convex_wrap(data):
        n = min(max(10, len(data) // 5), 2000)
        pts = [(i, (i * 37) % 1000) for i in range(n)]
        return convex_hull(pts)

    def closest_wrap(data):
        n = min(max(10, len(data) // 5), 500)
        pts = [(i, (i * 23) % 1000) for i in range(n)]
        return closest_pair(pts)

    return {"convex_hull": convex_wrap, "closest_pair": closest_wrap}


def _make_backtracking_wrappers():
    def nqueens_wrap(data):
        n = max(4, min(10, max(4, len(data) // 50)))
        return solve_n_queens(n)

    return {"n_queens": nqueens_wrap}


def _make_greedy_wrappers():
    def coin_change_wrap(data):
        coins = [1, 5, 10, 25]
        amount = max(1, len(data) * 2)
        return coin_change_greedy(coins, amount)

    return {"coin_change": coin_change_wrap}


def _make_ds_wrappers():
    def stack_ops(data):
        s = Stack()
        for x in data:
            s.push(x)
        while not s.is_empty():
            s.pop()
        return True

    def queue_ops(data):
        q = Queue()
        for x in data:
            q.enqueue(x)
        while not q.is_empty():
            q.dequeue()
        return True

    def linked_list_ops(data):
        ll = LinkedList()
        items = data[:5000]  # cap: append is O(n), so full list is O(n²)
        for x in items:
            ll.append(x)
        if items:
            return ll.find(items[-1])
        return -1

    def union_find_ops(data):
        n = max(2, min(1000, len(data)))
        uf = UnionFind(n)
        for i in range(1, n):
            uf.union(i - 1, i)
        return uf.find(0)

    def trie_ops(data):
        t = Trie()
        items = data[:10000]  # cap: pure-Python is slow at 100k strings
        for x in items:
            t.insert(str(x))
        if items:
            return t.search(str(items[-1]))
        return False

    def skiplist_ops(data):
        sl = SkipList()
        items = data[:10000]  # cap: pure-Python O(n log n) is slow at 100k
        for x in items:
            sl.insert(x)
        if items:
            return sl.search(items[-1])
        return False

    def bst_ops(data):
        b = BST()
        items = data[:10000]  # cap: pure-Python O(n log n) is slow at 100k
        for x in items:
            b.insert(x)
        if items:
            return b.contains(items[-1])
        return False

    def avl_ops(data):
        root = None
        items = data[:50000]  # cap: AVL impl has edge-case bug beyond ~50k items
        for x in items:
            root = avl_insert(root, x)
        return root is not None

    def bloom_ops(data):
        bf = BloomFilter()
        for x in data:
            bf.add(x)
        if data:
            return (data[-1] in bf)
        return False

    def heap_ops(data):
        h = MinHeap()
        for x in data:
            h.push(x)
        if data:
            return h.pop()
        return None

    def dll_ops(data):
        dll = DoublyLinkedList()
        for x in data:
            dll.append(x)
        if data:
            return dll.find(data[-1])
        return -1

    return {
        "stack_ops": stack_ops,
        "queue_ops": queue_ops,
        "linked_list_ops": linked_list_ops,
        "union_find_ops": union_find_ops,
        "trie_ops": trie_ops,
        "skiplist_ops": skiplist_ops,
        "bst_ops": bst_ops,
        "avl_ops": avl_ops,
        "bloom_ops": bloom_ops,
        "heap_ops": heap_ops,
        "dll_ops": dll_ops,
    }


GROUPS.update({
    "dynamic": {"cpu": _make_dynamic_wrappers(), "gpu": {}},
    "strings": {"cpu": _make_string_wrappers(), "gpu": {}},
    "math": {"cpu": _make_math_wrappers(), "gpu": {}},
    "geometry": {"cpu": _make_geometry_wrappers(), "gpu": {}},
    "backtracking": {"cpu": _make_backtracking_wrappers(), "gpu": {}},
    "greedy": {"cpu": _make_greedy_wrappers(), "gpu": {}},
    "data_structures": {"cpu": _make_ds_wrappers(), "gpu": {}},
})


def _build_tridiag_csr(n):
    """Return (data, indices, indptr, x) for an n×n tridiagonal CSR matrix."""
    data_list, indices, indptr = [], [], [0]
    for i in range(n):
        data_list.append(1.0)
        indices.append(i)
        if i + 1 < n:
            data_list.append(1.0)
            indices.append(i + 1)
        indptr.append(len(data_list))
    return data_list, indices, indptr, [1.0] * n


def _make_sparse_cpu_wrappers():
    """Pure-CPU (NumPy) CSR SpMV — reference baseline."""
    import numpy as _np

    def spmv_cpu(data):
        n = max(2, min(500, len(data) or 4))
        d, idx, ptr, x = _build_tridiag_csr(n)
        d_arr = _np.array(d, dtype=_np.float32)
        idx_arr = _np.array(idx, dtype=_np.int32)
        ptr_arr = _np.array(ptr, dtype=_np.int32)
        x_arr = _np.array(x, dtype=_np.float32)
        out = _np.zeros(n, dtype=_np.float32)
        for row in range(n):
            for j in range(ptr_arr[row], ptr_arr[row + 1]):
                out[row] += d_arr[j] * x_arr[idx_arr[j]]
        return out

    return {"spmv": spmv_cpu}


def _make_sparse_gpu_wrappers():
    """GPU-accelerated CSR SpMV (falls back to CPU when no GPU is present)."""
    def spmv_gpu(data):
        n = max(2, min(500, len(data) or 4))
        d, idx, ptr, x = _build_tridiag_csr(n)
        return gpu_spmv(d, idx, ptr, x)

    return {"spmv": spmv_gpu}


def _gpu_available():
    try:
        import importlib
        spec = importlib.util.find_spec('cupy')
        if spec is not None:
            try:
                import cupy as cp
                return cp.cuda.runtime.getDeviceCount() > 0
            except Exception:
                pass
    except Exception:
        pass
    try:
        from numba import cuda
        return getattr(cuda, 'is_available', lambda: False)()
    except Exception:
        return False


# Register additional CPU sorts and optional GPU kernels when available
_GPU = _gpu_available()
GROUPS['sort']['cpu'].update({
    'insertion': insertion_sort,
    'selection': selection_sort,
    'shell': shell_sort,
    # counting sort: use range proportional to n so benchmark measures O(n) not O(value_range)
    'counting': lambda data: counting_sort([x % (len(data) * 10) for x in data]),
    'radix': radix_sort,
})
if 'gpu' not in GROUPS['sort']:
    GROUPS['sort']['gpu'] = {}
GROUPS['sort']['gpu'].update({
    'gpu_sort': parallel_sort,
    'radix_cupy': radix_sort_cupy if _GPU else None,
    'cupy_sort': cupy_sort if _GPU else None,
})

# expose additional GPU math kernels when hardware/lib present
import numpy as _np

def _make_gpu_math_wrappers():
    def fft_wrap(data):
        return gpu_fft(data)

    def convolve_wrap(data):
        n = len(data)
        kernel = data[:max(1, n // 8)]  # use first 1/8 as filter
        return gpu_convolve(data, kernel)

    def prefix_sum_wrap(data):
        return gpu_prefix_sum(_np.array(data, dtype=_np.float32))

    def sum_reduce_wrap(data):
        return gpu_sum_reduce(_np.array(data, dtype=_np.float32))

    def max_reduce_wrap(data):
        return gpu_max_reduce(_np.array(data, dtype=_np.float32))

    def vec_add_wrap(data):
        a = _np.array(data, dtype=_np.float32)
        return gpu_vec_add(a, a)

    def matmul_numba_wrap(data):
        side = max(2, int(len(data) ** 0.5))
        A = _np.array(data[:side * side], dtype=_np.float32).reshape(side, side)
        return gpu_matmul_numba(A, A)

    def matmul_cupy_wrap(data):
        side = max(2, int(len(data) ** 0.5))
        A = _np.array(data[:side * side], dtype=_np.float32).reshape(side, side)
        return gpu_matmul_cupy(A, A)

    def min_reduce_wrap(data):
        return gpu_min_reduce(_np.array(data, dtype=_np.float32))

    def mean_reduce_wrap(data):
        return gpu_mean_reduce(_np.array(data, dtype=_np.float32))

    def dot_product_wrap(data):
        return gpu_dot_product(_np.array(data, dtype=_np.float32))

    def diff_wrap(data):
        return gpu_diff(_np.array(data, dtype=_np.float32))

    def argsort_wrap(data):
        return argsort_gpu(_np.array(data, dtype=_np.float32))

    def clip_wrap(data):
        return clip_gpu(_np.array(data, dtype=_np.float32))

    def histogram_wrap(data):
        return histogram_gpu(_np.array(data, dtype=_np.float32))

    def softmax_wrap(data):
        return softmax_gpu(_np.array(data, dtype=_np.float32))

    def norm_wrap(data):
        return gpu_norm(_np.array(data, dtype=_np.float32))

    def outer_product_wrap(data):
        # cap at 1000 elements to keep outer product O(n) in benchmark time
        return gpu_outer_product(_np.array(data[:1000], dtype=_np.float32))

    result = {}
    if gpu_fft:
        result['fft'] = fft_wrap
    if gpu_convolve:
        result['convolve'] = convolve_wrap
    if gpu_prefix_sum:
        result['prefix_sum'] = prefix_sum_wrap
    if gpu_sum_reduce:
        result['sum_reduce'] = sum_reduce_wrap
    if gpu_max_reduce:
        result['max_reduce'] = max_reduce_wrap
    if gpu_vec_add and _numba_cuda_available:
        result['vec_add'] = vec_add_wrap
    if gpu_matmul_numba and _numba_cuda_available:
        result['matmul_numba'] = matmul_numba_wrap
    if gpu_matmul_cupy:
        result['matmul_cupy'] = matmul_cupy_wrap
    if gpu_min_reduce:
        result['min_reduce'] = min_reduce_wrap
    if gpu_mean_reduce:
        result['mean_reduce'] = mean_reduce_wrap
    if gpu_dot_product:
        result['dot_product'] = dot_product_wrap
    if gpu_diff:
        result['diff'] = diff_wrap
    if argsort_gpu:
        result['argsort'] = argsort_wrap
    if clip_gpu:
        result['clip'] = clip_wrap
    if histogram_gpu:
        result['histogram'] = histogram_wrap
    if softmax_gpu:
        result['softmax'] = softmax_wrap

    def matmul_tiled_wrap(data):
        side = max(2, int(len(data) ** 0.5))
        A = _np.array(data[:side * side], dtype=_np.float32).reshape(side, side)
        return gpu_matmul_tiled(A, A)

    if gpu_norm:
        result['norm'] = norm_wrap
    if gpu_outer_product:
        result['outer_product'] = outer_product_wrap
    if gpu_matmul_tiled and _numba_cuda_available:
        result['matmul_tiled'] = matmul_tiled_wrap
    return result

if 'math' in GROUPS and _GPU:
    GROUPS['math']['gpu'].update(_make_gpu_math_wrappers())

# add sparse group with separate CPU (numpy) and GPU (sparse_ops) backends
GROUPS.update({
    'sparse': {
        'cpu': _make_sparse_cpu_wrappers(),
        'gpu': _make_sparse_gpu_wrappers() if gpu_spmv is not None else {},
    }
})


def run_one(func, data, repeat: int, include_memory: bool):
    wrapped = benchmark(func, include_memory=include_memory, repeat=repeat)
    wrapped(list(data))
    return wrapped.last_metrics

DEFAULT_FIELDNAMES = ["algorithm", "backend", "n", "runs", "min", "max", "mean", "p50", "p95", "times"]
SLOW_ALGOS = {}  # Empty set: run all algorithms at all sizes (no skipping)
# WARNING: bubble/insertion/selection are O(n²) and running them at n=100000 will be extremely slow!
SLOW_ALGOS_MAX_N = 100000


def merge(inputs_glob, out_path):
    files = sorted(glob.glob(inputs_glob))
    if not files:
        print('No files matched', inputs_glob)
        return 1
    found_fields = []
    rows = []
    for f in files:
        with open(f, newline='') as fh:
            r = csv.DictReader(fh)
            fn = r.fieldnames or []
            for name in fn:
                if name not in found_fields:
                    found_fields.append(name)
            for row in r:
                row['_source'] = Path(f).name
                rows.append(row)
    # canonical field order: DEFAULT_FIELDNAMES first, then any others
    fieldnames = [f for f in DEFAULT_FIELDNAMES if f in found_fields] + [f for f in found_fields if f not in DEFAULT_FIELDNAMES]
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            out_row = {k: row.get(k, '') for k in fieldnames}
            w.writerow(out_row)
    print('Wrote', out, 'with', len(rows), 'rows (from', len(files), 'files)')
    return 0


def run_full(sizes, repeat, out, include_memory):
    results = []
    groups = GROUPS
    for group_name, group in groups.items():
        for backend, algos in group.items():
            for name, func in algos.items():
                if func is None:
                    continue
                # warmup GPU functions once per function to absorb JIT / driver init overhead
                if backend == 'gpu':
                    try:
                        func(list(random_list(min(sizes[0] if sizes else 100, 1000))))
                    except Exception:
                        pass
                for n in sizes:
                    if name in SLOW_ALGOS and n > SLOW_ALGOS_MAX_N:
                        print(f"Skipping {name} for n={n} (too slow)")
                        continue
                    data = random_list(n)
                    try:
                        metrics = run_one(func, data, repeat, include_memory)
                    except Exception as e:
                        print(f"{name} ({backend}) failed for n={n}: {e}")
                        continue
                    entry = {"algorithm": name, "backend": backend, "n": n, **metrics}
                    results.append(entry)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = DEFAULT_FIELDNAMES
    with out_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {k: r.get(k) for k in fieldnames}
            row["times"] = ";".join(str(t) for t in r.get("times", []))
            writer.writerow(row)
    print("Full benchmark run complete:", out_path)


def main():
    parser = argparse.ArgumentParser(description="Run benchmarks for algorithm groups, or run full sweeps, or merge CSVs")
    parser.add_argument("--group", choices=list(GROUPS.keys()), default="sort")
    parser.add_argument("--sizes", nargs="+", type=int, default=[100, 1000])
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--out", type=str, default="benchmarks/results.csv")
    parser.add_argument("--include-memory", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--full", action="store_true", help="Run safe full sweep and write results to --out (overrides --group)")
    parser.add_argument("--merge-inputs", type=str, default=None, help='glob pattern for input CSVs to merge')
    parser.add_argument("--merge-out", type=str, default="benchmarks/results_merged.csv", help='output path when merging')
    args = parser.parse_args()

    if args.merge_inputs:
        return merge(args.merge_inputs, args.merge_out)

    if args.full:
        return run_full(args.sizes, args.repeat, args.out, args.include_memory)

    results = []
    group = GROUPS[args.group]
    for backend, algos in group.items():
        for name, func in algos.items():
            if func is None:
                continue
            for n in args.sizes:
                data = random_list(n)
                try:
                    metrics = run_one(func, data, args.repeat, args.include_memory)
                except Exception as e:
                    print(f"{name} ({backend}) failed for n={n}: {e}")
                    continue
                entry = {"algorithm": name, "backend": backend, "n": n, **metrics}
                results.append(entry)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = DEFAULT_FIELDNAMES
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {k: r.get(k) for k in fieldnames}
            row["times"] = ";".join(str(t) for t in r.get("times", []))
            writer.writerow(row)

    if args.plot:
        print("PNG saving has been disabled. Use the Dash dashboard for visualization.")


if __name__ == "__main__":
    if not _acquire_lock():
        print(
            f"[run_benchmarks] Another instance is already running "
            f"(lock: {_LOCK_PATH}). Exiting.",
            file=sys.stderr,
        )
        sys.exit(1)
    main()
