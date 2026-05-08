from algorithms.cpu.dynamic_prog.fib_memo import fib
from algorithms.cpu.dynamic_prog.knapsack import knapsack_01


# ── Fibonacci ────────────────────────────────────────────────────────────────

def test_fib_base_cases():
    assert fib(0) == 0
    assert fib(1) == 1


def test_fib_known_values():
    expected = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
    for n, val in enumerate(expected):
        assert fib(n) == val, f"fib({n}) expected {val}"


def test_fib_large():
    # Verify memoisation handles large n without stack overflow
    assert fib(100) == 354224848179261915075


def test_fib_independent_calls_share_no_state():
    # Each call with a fresh memo must still be correct
    assert fib(10) == 55
    assert fib(10) == 55


# ── 0/1 Knapsack ─────────────────────────────────────────────────────────────

def test_knapsack_zero_capacity():
    assert knapsack_01([1, 2], [10, 20], 0) == 0


def test_knapsack_no_items():
    assert knapsack_01([], [], 10) == 0


def test_knapsack_single_item_fits():
    assert knapsack_01([3], [7], 5) == 7


def test_knapsack_single_item_too_heavy():
    assert knapsack_01([6], [7], 5) == 0


def test_knapsack_classic():
    # weights=[2,3,4,5], values=[3,4,5,6], capacity=5
    # Best: items 0+1 → weight=5, value=7
    assert knapsack_01([2, 3, 4, 5], [3, 4, 5, 6], 5) == 7


def test_knapsack_all_items_fit():
    weights = [1, 1, 1]
    values = [10, 20, 30]
    assert knapsack_01(weights, values, 10) == 60


def test_knapsack_each_item_once():
    # Classic 0/1 — each item used at most once
    weights = [3, 3]
    values = [5, 5]
    # Capacity 4: can only take one item (weight 3 each)
    assert knapsack_01(weights, values, 4) == 5


if __name__ == "__main__":
    test_fib_base_cases()
    test_fib_known_values()
    test_fib_large()
    test_fib_independent_calls_share_no_state()
    test_knapsack_zero_capacity()
    test_knapsack_no_items()
    test_knapsack_single_item_fits()
    test_knapsack_single_item_too_heavy()
    test_knapsack_classic()
    test_knapsack_all_items_fit()
    test_knapsack_each_item_once()
    print("Dynamic programming tests passed")
