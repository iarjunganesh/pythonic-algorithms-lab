from algorithms.cpu.math.gcd import gcd
from algorithms.cpu.math.sieve import sieve


# ── GCD ──────────────────────────────────────────────────────────────────────

def test_gcd_basic():
    assert gcd(12, 8) == 4
    assert gcd(7, 5) == 1
    assert gcd(100, 25) == 25


def test_gcd_same_number():
    assert gcd(9, 9) == 9


def test_gcd_one():
    assert gcd(1, 99) == 1
    assert gcd(99, 1) == 1


def test_gcd_commutative():
    assert gcd(48, 18) == gcd(18, 48)


def test_gcd_with_zero():
    assert gcd(0, 5) == 5
    assert gcd(5, 0) == 5


def test_gcd_negative_inputs():
    # gcd is defined as non-negative
    assert gcd(-12, 8) == 4
    assert gcd(12, -8) == 4
    assert gcd(-12, -8) == 4


# ── Sieve of Eratosthenes ────────────────────────────────────────────────────

def test_sieve_below_two():
    assert sieve(0) == []
    assert sieve(1) == []


def test_sieve_small():
    assert sieve(2) == [2]
    assert sieve(10) == [2, 3, 5, 7]
    assert sieve(20) == [2, 3, 5, 7, 11, 13, 17, 19]


def test_sieve_includes_upper_bound():
    primes = sieve(11)
    assert 11 in primes


def test_sieve_excludes_composites():
    primes = sieve(30)
    for p in primes:
        assert all(p % d != 0 for d in range(2, p)), f"{p} is not prime"


def test_sieve_count():
    # There are 25 primes below 100
    assert len(sieve(100)) == 25


if __name__ == "__main__":
    test_gcd_basic()
    test_gcd_same_number()
    test_gcd_one()
    test_gcd_commutative()
    test_gcd_with_zero()
    test_gcd_negative_inputs()
    test_sieve_below_two()
    test_sieve_small()
    test_sieve_includes_upper_bound()
    test_sieve_excludes_composites()
    test_sieve_count()
    print("Math tests passed")
