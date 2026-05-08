import random
from typing import List

try:
    import numpy as np
except Exception:
    np = None


def random_list(n: int, low: int = 0, high: int = 1_000_000) -> List[int]:
    return [random.randint(low, high) for _ in range(n)]


def random_array(n: int, low: int = 0, high: int = 1_000_000, dtype=int):
    if np is None:
        raise RuntimeError("numpy is required for random_array")
    return np.random.randint(low, high, size=n, dtype=dtype)


def sorted_list(n: int) -> List[int]:
    return list(range(n))


def reversed_list(n: int) -> List[int]:
    return list(range(n - 1, -1, -1))
