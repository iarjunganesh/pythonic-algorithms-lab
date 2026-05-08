from collections import deque


class Queue:
    def __init__(self):
        self._q = deque()

    def enqueue(self, x):
        self._q.append(x)

    def dequeue(self):
        return self._q.popleft()

    def is_empty(self):
        return not self._q
