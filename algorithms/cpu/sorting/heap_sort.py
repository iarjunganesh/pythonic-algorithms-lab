import heapq


def heap_sort(arr):
    a = list(arr)
    heapq.heapify(a)
    out = [heapq.heappop(a) for _ in range(len(a))]
    return out
