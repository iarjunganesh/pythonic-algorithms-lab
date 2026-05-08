"""Singly linked list implementation."""

class Node:
    def __init__(self, value):
        self.value = value
        self.next = None


class LinkedList:
    def __init__(self, iterable=None):
        self.head = None
        if iterable:
            for v in iterable:
                self.append(v)

    def append(self, value):
        node = Node(value)
        if not self.head:
            self.head = node
            return
        cur = self.head
        while cur.next:
            cur = cur.next
        cur.next = node

    def to_list(self):
        out = []
        cur = self.head
        while cur:
            out.append(cur.value)
            cur = cur.next
        return out

    def find(self, value):
        idx = 0
        cur = self.head
        while cur:
            if cur.value == value:
                return idx
            cur = cur.next
            idx += 1
        return -1

    def remove(self, value):
        if not self.head:
            return False
        if self.head.value == value:
            self.head = self.head.next
            return True
        prev = self.head
        cur = prev.next
        while cur:
            if cur.value == value:
                prev.next = cur.next
                return True
            prev = cur
            cur = cur.next
        return False
