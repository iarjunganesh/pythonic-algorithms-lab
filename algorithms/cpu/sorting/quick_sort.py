def quick_sort(arr):
    """Sort a list using recursive quicksort with middle-element pivot.

    The middle-element pivot avoids O(n²) behaviour on already-sorted input.
    Note: Python's default recursion limit (~1000) caps practical input size
    to roughly 5 000–10 000 elements before hitting RecursionError.
    For larger inputs, use tim_sort or merge_sort instead.
    """
    a = list(arr)
    if len(a) <= 1:
        return a
    pivot = a[len(a) // 2]
    lo = [x for x in a if x < pivot]
    eq = [x for x in a if x == pivot]
    hi = [x for x in a if x > pivot]
    return quick_sort(lo) + eq + quick_sort(hi)
