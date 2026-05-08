def suffix_array(s: str):
    """Naive suffix array construction (suitable for small/mid-size strings)."""
    return sorted(range(len(s)), key=lambda i: s[i:])
