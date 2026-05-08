def fibonacci_search(arr, target):
    """Fibonacci search on sorted array. Returns index or -1."""
    n = len(arr)
    if n == 0:
        return -1
    fibMm2 = 0  # (m-2)'th Fibonacci No.
    fibMm1 = 1  # (m-1)'th Fibonacci No.
    fibM = fibMm1 + fibMm2  # m'th Fibonacci
    while fibM < n:
        fibMm2 = fibMm1
        fibMm1 = fibM
        fibM = fibMm1 + fibMm2
    offset = -1
    while fibM > 1:
        i = min(offset + fibMm2, n - 1)
        if arr[i] < target:
            fibM = fibMm1
            fibMm1 = fibMm2
            fibMm2 = fibM - fibMm1
            offset = i
        elif arr[i] > target:
            fibM = fibMm2
            fibMm1 = fibMm1 - fibMm2
            fibMm2 = fibM - fibMm1
        else:
            return i
    if fibMm1 and offset + 1 < n and arr[offset + 1] == target:
        return offset + 1
    return -1
