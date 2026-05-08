def counting_sort(arr):
    """Counting sort for integers (handles negatives by offset)."""
    if not arr:
        return []

    # Counting sort only makes sense for integers; fall back otherwise.
    if not all(isinstance(x, int) for x in arr):
        return sorted(arr)

    amin = min(arr)
    amax = max(arr)
    offset = -amin
    size = amax - amin + 1

    # Avoid huge memory allocation for pathological ranges — fall back.
    if size > 10_000_000:
        return sorted(arr)

    counts = [0] * size
    for x in arr:
        counts[x + offset] += 1
    out = []
    for i, c in enumerate(counts):
        out.extend([i - offset] * c)
    return out
