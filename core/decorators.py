import time
import tracemalloc
import functools
from typing import Callable, Any, Dict


def benchmark(func: Callable = None, *, include_memory: bool = False, repeat: int = 1):
    """Decorator to measure execution time (and optional peak memory).

    Usage:
        @benchmark
        def f(...):
            ...

        @benchmark(repeat=3, include_memory=True)
        def g(...):
            ...

    After a call, metrics are available on `wrapped.last_metrics`.
    """

    if func is None:
        return lambda f: benchmark(f, include_memory=include_memory, repeat=repeat)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        times = []
        mem_peaks = []
        result = None
        for _ in range(max(1, repeat)):
            if include_memory:
                tracemalloc.start()
            t0 = time.perf_counter()
            result = func(*args, **kwargs)
            t1 = time.perf_counter()
            if include_memory:
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                mem_peaks.append(peak)
            times.append(t1 - t0)

        sorted_times = sorted(times)
        n = len(sorted_times)

        def _pct(p):
            idx = (p / 100) * (n - 1)
            lo, hi = int(idx), min(int(idx) + 1, n - 1)
            return sorted_times[lo] + (idx - lo) * (sorted_times[hi] - sorted_times[lo])

        metrics: Dict[str, Any] = {
            "runs": len(times),
            "min": min(times),
            "max": max(times),
            "mean": sum(times) / len(times),
            "p50": _pct(50),
            "p95": _pct(95),
            "times": times,
        }
        if include_memory:
            metrics["peak_mem_bytes"] = max(mem_peaks) if mem_peaks else None

        wrapper.last_metrics = metrics
        return result

    wrapper.last_metrics = None
    return wrapper
