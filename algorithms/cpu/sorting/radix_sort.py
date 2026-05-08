def radix_sort(arr):
    """LSD Radix sort for integers (handles negatives)."""
    if not arr:
        return []
    negatives = [x for x in arr if x < 0]
    positives = [x for x in arr if x >= 0]

    def _radix(a):
        if not a:
            return []
        out = list(a)
        maxv = max(out)
        exp = 1
        while maxv // exp > 0:
            buckets = [[] for _ in range(10)]
            for num in out:
                buckets[(num // exp) % 10].append(num)
            out = [v for bucket in buckets for v in bucket]
            exp *= 10
        return out

    pos_sorted = _radix(positives)
    neg_sorted = _radix([abs(x) for x in negatives])
    neg_sorted = [-x for x in reversed(neg_sorted)]
    return neg_sorted + pos_sorted
