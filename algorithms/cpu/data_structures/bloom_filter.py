"""Simple Bloom filter implementation using multiple hash salts."""

class BloomFilter:
    def __init__(self, m=1024, k=3):
        self.m = m
        self.k = k
        self.bitset = 0

    def _indices(self, item):
        for i in range(self.k):
            h = hash((item, i))
            yield abs(h) % self.m

    def add(self, item):
        for idx in self._indices(item):
            self.bitset |= 1 << idx

    def __contains__(self, item):
        for idx in self._indices(item):
            if not (self.bitset >> idx) & 1:
                return False
        return True
