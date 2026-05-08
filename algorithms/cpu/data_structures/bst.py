"""Binary Search Tree (BST) with insert/search and inorder traversal."""

class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None


class BST:
    def __init__(self):
        self.root = None

    def insert(self, value):
        def _insert(node, value):
            if node is None:
                return Node(value)
            if value < node.value:
                node.left = _insert(node.left, value)
            else:
                node.right = _insert(node.right, value)
            return node

        self.root = _insert(self.root, value)

    def contains(self, value):
        cur = self.root
        while cur:
            if cur.value == value:
                return True
            cur = cur.left if value < cur.value else cur.right
        return False

    def inorder(self):
        def _rec(node):
            if node is None:
                return []
            return _rec(node.left) + [node.value] + _rec(node.right)

        return _rec(self.root)
