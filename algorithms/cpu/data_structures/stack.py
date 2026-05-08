class Stack:
    def __init__(self):
        self._data = []

    def push(self, x):
        self._data.append(x)

    def pop(self):
        return self._data.pop()

    def top(self):
        return self._data[-1] if self._data else None

    def is_empty(self):
        return not self._data
