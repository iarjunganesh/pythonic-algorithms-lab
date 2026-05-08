from algorithms.cpu.greedy.coin_change import coin_change_greedy


def test_exact_change():
    result = coin_change_greedy([1, 5, 10, 25], 41)
    assert result is not None
    assert sum(result) == 41


def test_zero_amount():
    result = coin_change_greedy([1, 5, 10], 0)
    assert result == []


def test_single_coin():
    result = coin_change_greedy([5], 15)
    assert result == [5, 5, 5]


def test_canonical_system_optimal():
    # US coins: greedy is optimal for canonical coin sets
    result = coin_change_greedy([1, 5, 10, 25], 30)
    assert sorted(result, reverse=True) == [25, 5]
    assert sum(result) == 30


def test_impossible_change():
    # No way to make 3 cents with only 5-cent and 10-cent coins
    result = coin_change_greedy([5, 10], 3)
    assert result is None


def test_large_denomination_first():
    result = coin_change_greedy([1, 5, 10, 25], 99)
    assert result is not None
    assert sum(result) == 99
    # Greedy should use largest denominations first
    assert 25 in result


if __name__ == "__main__":
    test_exact_change()
    test_zero_amount()
    test_single_coin()
    test_canonical_system_optimal()
    test_impossible_change()
    test_large_denomination_first()
    print("Greedy tests passed")
