from algorithms.cpu.geometry.convex_hull import convex_hull
from algorithms.cpu.geometry.closest_pair import closest_pair


def test_convex_hull_square():
    square = [(0, 0), (1, 0), (1, 1), (0, 1)]
    hull = convex_hull(square)
    assert set(hull) == set(square)


def test_closest_pair():
    points = [(0, 0), (10, 10), (0.1, 0.1), (5, 5)]
    pair, d = closest_pair(points)
    assert pair is not None
    assert d > 0


if __name__ == "__main__":
    test_convex_hull_square()
    test_closest_pair()
    print("Geometry tests passed")
