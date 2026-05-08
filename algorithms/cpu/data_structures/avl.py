"""AVL (self-balancing) binary search tree implementation."""

class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
        self.height = 1


def _height(node):
    return node.height if node else 0


def _update_height(node):
    node.height = 1 + max(_height(node.left), _height(node.right))


def _balance_factor(node):
    return _height(node.left) - _height(node.right)


def _rotate_right(y):
    x = y.left
    T2 = x.right
    x.right = y
    y.left = T2
    _update_height(y)
    _update_height(x)
    return x


def _rotate_left(x):
    y = x.right
    T2 = y.left
    y.left = x
    x.right = T2
    _update_height(x)
    _update_height(y)
    return y


def insert(root, value):
    if root is None:
        return Node(value)
    if value < root.value:
        root.left = insert(root.left, value)
    else:
        root.right = insert(root.right, value)

    _update_height(root)
    balance = _balance_factor(root)

    if balance > 1 and value < root.left.value:
        return _rotate_right(root)
    if balance < -1 and value > root.right.value:
        return _rotate_left(root)
    if balance > 1 and value > root.left.value:
        root.left = _rotate_left(root.left)
        return _rotate_right(root)
    if balance < -1 and value < root.right.value:
        root.right = _rotate_right(root.right)
        return _rotate_left(root)

    return root
