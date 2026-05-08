"""Closest pair (brute-force) for small point sets."""
import math


def closest_pair(points):
    """Return ((p1, p2), dist) where dist is Euclidean distance."""
    pts = list(points)
    n = len(pts)
    if n < 2:
        return (None, None), float('inf')
    best = float('inf')
    pair = (None, None)
    for i in range(n):
        for j in range(i + 1, n):
            dx = pts[i][0] - pts[j][0]
            dy = pts[i][1] - pts[j][1]
            d = math.hypot(dx, dy)
            if d < best:
                best = d
                pair = (pts[i], pts[j])
    return pair, best
