def coin_change_greedy(coins, amount):
    """Greedy coin change for canonical coin systems (returns list of coins)."""
    coins = sorted(coins, reverse=True)
    res = []
    for c in coins:
        while amount >= c:
            amount -= c
            res.append(c)
    if amount != 0:
        return None
    return res
