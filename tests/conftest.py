import random
import pytest


@pytest.fixture
def random_ints():
    """Return a seeded list of 50 random integers in [-100, 100]."""
    random.seed(42)
    return [random.randint(-100, 100) for _ in range(50)]


@pytest.fixture
def sorted_ints():
    return list(range(50))


@pytest.fixture
def reversed_ints():
    return list(range(49, -1, -1))
