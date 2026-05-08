from algorithms.cpu.searching.binary_search import binary_search
from algorithms.cpu.searching.linear_search import linear_search
from algorithms.cpu.searching.jump_search import jump_search
from algorithms.cpu.searching.fibonacci_search import fibonacci_search


def _check_search(func):
    arr = list(range(100))
    for i in range(100):
        assert func(arr, i) == i
    assert func(arr, -1) == -1


def test_binary_search():
    _check_search(binary_search)


def test_linear_search():
    _check_search(linear_search)


def test_jump_search():
    _check_search(jump_search)


def test_fibonacci_search():
    _check_search(fibonacci_search)


if __name__ == "__main__":
    test_binary_search()
    test_linear_search()
    test_jump_search()
    test_fibonacci_search()
    print("Searching tests passed")
