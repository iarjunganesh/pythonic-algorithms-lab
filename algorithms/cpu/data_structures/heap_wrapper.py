import heapq


class MinHeap:
    def __init__(self):
        self._h = []

    def push(self, x):
        heapq.heappush(self._h, x)

    def pop(self):
        return heapq.heappop(self._h)

    def peek(self):
        return self._h[0] if self._h else None
