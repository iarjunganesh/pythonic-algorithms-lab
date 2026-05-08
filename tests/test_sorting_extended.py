import random

from algorithms.cpu.sorting.insertion_sort import insertion_sort
from algorithms.cpu.sorting.selection_sort import selection_sort
from algorithms.cpu.sorting.quick_sort import quick_sort
from algorithms.cpu.sorting.heap_sort import heap_sort
from algorithms.cpu.sorting.shell_sort import shell_sort
from algorithms.cpu.sorting.counting_sort import counting_sort
from algorithms.cpu.sorting.radix_sort import radix_sort
from algorithms.cpu.sorting.tim_sort import tim_sort
from algorithms.cpu.sorting.merge_sort import merge_sort
from algorithms.cpu.sorting.bubble_sort import bubble_sort


SORT_FUNCS = [
    ("insertion", insertion_sort),
    ("selection", selection_sort),
    ("quick", quick_sort),
    ("heap", heap_sort),
    ("shell", shell_sort),
    ("counting", counting_sort),
    ("radix", radix_sort),
    ("timsort", tim_sort),
    ("merge", merge_sort),
    ("bubble", bubble_sort),
]


def _check_equal(arr):
    expected = sorted(arr)
    for name, func in SORT_FUNCS:
        out = func(arr)
        assert out == expected, f"{name} failed: {out} != {expected}"


def test_basic_cases():
    cases = [[], [1], [2, 1], [3, 1, 2], [5, -1, 3, 0], list(range(10))[::-1]]
    for c in cases:
        _check_equal(c)


def test_random_cases():
    random.seed(0)
    for _ in range(5):
        arr = [random.randint(-100, 100) for _ in range(50)]
        _check_equal(arr)


if __name__ == "__main__":
    test_basic_cases()
    test_random_cases()
    print("Extended sorting tests passed")
