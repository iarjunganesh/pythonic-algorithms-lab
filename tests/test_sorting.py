from algorithms.cpu.sorting.bubble_sort import bubble_sort
from algorithms.cpu.sorting.merge_sort import merge_sort


def test_bubble_sort_empty():
    assert bubble_sort([]) == []


def test_bubble_sort_sorted():
    assert bubble_sort([1, 2, 3]) == [1, 2, 3]


def test_merge_sort_random():
    assert merge_sort([3, 1, 2]) == [1, 2, 3]

if __name__ == '__main__':
    test_bubble_sort_empty()
    test_bubble_sort_sorted()
    test_merge_sort_random()
    print('All tests passed!')
