def rabin_karp(text: str, pattern: str) -> int:
    """Return first index of pattern in text or -1 (Rabin-Karp)."""
    if pattern == "":
        return 0
    n, m = len(text), len(pattern)
    if m > n:
        return -1
    base = 256
    mod = 101
    h = 1
    for _ in range(m - 1):
        h = (h * base) % mod

    p_hash = 0
    t_hash = 0
    for i in range(m):
        p_hash = (base * p_hash + ord(pattern[i])) % mod
        t_hash = (base * t_hash + ord(text[i])) % mod

    for i in range(n - m + 1):
        if p_hash == t_hash:
            if text[i : i + m] == pattern:
                return i
        if i < n - m:
            t_hash = (base * (t_hash - ord(text[i]) * h) + ord(text[i + m])) % mod
            if t_hash < 0:
                t_hash += mod
    return -1
