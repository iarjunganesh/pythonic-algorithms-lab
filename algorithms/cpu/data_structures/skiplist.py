"""Simple SkipList implementation (probabilistic linked-list levels)."""

import random


class Node:
    def __init__(self, value, level):
        self.value = value
        self.forward = [None] * (level + 1)


class SkipList:
    MAX_LEVEL = 16
    P = 0.5

    def __init__(self):
        self.header = Node(None, self.MAX_LEVEL)
        self.level = 0

    def _random_level(self):
        lvl = 0
        while random.random() < self.P and lvl < self.MAX_LEVEL:
            lvl += 1
        return lvl

    def insert(self, value):
        update = [None] * (self.MAX_LEVEL + 1)
        cur = self.header
        for i in range(self.level, -1, -1):
            while cur.forward[i] and cur.forward[i].value < value:
                cur = cur.forward[i]
            update[i] = cur
        cur = cur.forward[0]
        if cur is None or cur.value != value:
            lvl = self._random_level()
            if lvl > self.level:
                for i in range(self.level + 1, lvl + 1):
                    update[i] = self.header
                self.level = lvl
            node = Node(value, lvl)
            for i in range(lvl + 1):
                node.forward[i] = update[i].forward[i]
                update[i].forward[i] = node

    def search(self, value):
        cur = self.header
        for i in range(self.level, -1, -1):
            while cur.forward[i] and cur.forward[i].value < value:
                cur = cur.forward[i]
        cur = cur.forward[0]
        return cur is not None and cur.value == value

    def to_list(self):
        out = []
        cur = self.header.forward[0]
        while cur:
            out.append(cur.value)
            cur = cur.forward[0]
        return out
