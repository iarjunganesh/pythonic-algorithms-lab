from algorithms.cpu.data_structures.linked_list import LinkedList
from algorithms.cpu.data_structures.doubly_linked_list import DoublyLinkedList
from algorithms.cpu.data_structures.bst import BST
from algorithms.cpu.data_structures.trie import Trie
from algorithms.cpu.data_structures.avl import insert as avl_insert
from algorithms.cpu.data_structures.bloom_filter import BloomFilter
from algorithms.cpu.data_structures.heap_wrapper import MinHeap
from algorithms.cpu.data_structures.skiplist import SkipList
from algorithms.cpu.data_structures.union_find import UnionFind
from algorithms.cpu.data_structures.stack import Stack
from algorithms.cpu.data_structures.queue import Queue


# ── Linked List ──────────────────────────────────────────────────────────────

def test_linked_list():
    ll = LinkedList([1, 2, 3])
    assert ll.to_list() == [1, 2, 3]
    assert ll.find(2) == 1
    assert ll.remove(2) is True
    assert ll.to_list() == [1, 3]


# ── Doubly Linked List ────────────────────────────────────────────────────────

def test_doubly_linked_list():
    dll = DoublyLinkedList([1, 2, 3])
    assert dll.to_list() == [1, 2, 3]
    dll.prepend(0)
    assert dll.to_list() == [0, 1, 2, 3]


# ── BST ──────────────────────────────────────────────────────────────────────

def test_bst():
    bst = BST()
    for v in [3, 1, 4, 2]:
        bst.insert(v)
    assert bst.inorder() == [1, 2, 3, 4]
    assert bst.contains(4)
    assert not bst.contains(99)


# ── AVL Tree ──────────────────────────────────────────────────────────────────

def _avl_inorder(node):
    if node is None:
        return []
    return _avl_inorder(node.left) + [node.value] + _avl_inorder(node.right)


def _avl_height(node):
    return node.height if node else 0


def _avl_balanced(node):
    if node is None:
        return True
    bf = _avl_height(node.left) - _avl_height(node.right)
    return abs(bf) <= 1 and _avl_balanced(node.left) and _avl_balanced(node.right)


def test_avl_inorder_sorted():
    values = [5, 3, 7, 1, 4, 6, 8]
    root = None
    for v in values:
        root = avl_insert(root, v)
    assert _avl_inorder(root) == sorted(values)


def test_avl_stays_balanced():
    # Insert ascending values — a plain BST would degenerate; AVL must stay balanced
    root = None
    for v in range(1, 16):
        root = avl_insert(root, v)
    assert _avl_balanced(root)


# ── Trie ──────────────────────────────────────────────────────────────────────

def test_trie():
    t = Trie()
    t.insert("hello")
    assert t.search("hello")
    assert not t.search("hell")
    assert t.starts_with("hell")


def test_trie_multiple_words():
    t = Trie()
    for word in ["apple", "app", "application"]:
        t.insert(word)
    assert t.search("app")
    assert t.search("apple")
    assert not t.search("ap")
    assert t.starts_with("appl")


# ── Bloom Filter ─────────────────────────────────────────────────────────────

def test_bloom_filter_membership():
    bf = BloomFilter(m=512, k=3)
    for item in ["alpha", "beta", "gamma"]:
        bf.add(item)
    assert "alpha" in bf
    assert "beta" in bf
    assert "gamma" in bf


def test_bloom_filter_no_false_negatives():
    bf = BloomFilter(m=1024, k=4)
    items = [f"item_{i}" for i in range(50)]
    for item in items:
        bf.add(item)
    for item in items:
        assert item in bf, f"{item} must be in filter after insertion"


# ── MinHeap ───────────────────────────────────────────────────────────────────

def test_minheap_push_pop():
    h = MinHeap()
    for v in [5, 3, 8, 1, 4]:
        h.push(v)
    assert h.pop() == 1
    assert h.pop() == 3
    assert h.pop() == 4


def test_minheap_peek():
    h = MinHeap()
    h.push(10)
    h.push(2)
    assert h.peek() == 2


def test_minheap_empty_peek():
    assert MinHeap().peek() is None


# ── SkipList ──────────────────────────────────────────────────────────────────

def test_skiplist_insert_search():
    sl = SkipList()
    for v in [3, 1, 4, 1, 5, 9]:
        sl.insert(v)
    assert sl.search(3)
    assert sl.search(9)
    assert not sl.search(7)


def test_skiplist_to_list_sorted():
    sl = SkipList()
    for v in [5, 3, 8, 1]:
        sl.insert(v)
    assert sl.to_list() == sorted({5, 3, 8, 1})


# ── UnionFind ────────────────────────────────────────────────────────────────

def test_union_find_basic():
    uf = UnionFind(5)
    assert uf.find(0) != uf.find(1)
    uf.union(0, 1)
    assert uf.find(0) == uf.find(1)


def test_union_find_transitive():
    uf = UnionFind(5)
    uf.union(0, 1)
    uf.union(1, 2)
    assert uf.find(0) == uf.find(2)


def test_union_find_already_connected():
    uf = UnionFind(3)
    assert uf.union(0, 1) is True
    assert uf.union(0, 1) is False  # already in same set


# ── Stack ────────────────────────────────────────────────────────────────────

def test_stack_push_pop():
    s = Stack()
    s.push(1)
    s.push(2)
    s.push(3)
    assert s.pop() == 3
    assert s.pop() == 2


def test_stack_top():
    s = Stack()
    s.push(42)
    assert s.top() == 42
    s.pop()
    assert s.top() is None


def test_stack_is_empty():
    s = Stack()
    assert s.is_empty()
    s.push(1)
    assert not s.is_empty()


# ── Queue ────────────────────────────────────────────────────────────────────

def test_queue_enqueue_dequeue():
    q = Queue()
    q.enqueue(1)
    q.enqueue(2)
    q.enqueue(3)
    assert q.dequeue() == 1
    assert q.dequeue() == 2


def test_queue_is_empty():
    q = Queue()
    assert q.is_empty()
    q.enqueue(5)
    assert not q.is_empty()


if __name__ == "__main__":
    test_linked_list()
    test_doubly_linked_list()
    test_bst()
    test_avl_inorder_sorted()
    test_avl_stays_balanced()
    test_trie()
    test_trie_multiple_words()
    test_bloom_filter_membership()
    test_bloom_filter_no_false_negatives()
    test_minheap_push_pop()
    test_minheap_peek()
    test_minheap_empty_peek()
    test_skiplist_insert_search()
    test_skiplist_to_list_sorted()
    test_union_find_basic()
    test_union_find_transitive()
    test_union_find_already_connected()
    test_stack_push_pop()
    test_stack_top()
    test_stack_is_empty()
    test_queue_enqueue_dequeue()
    test_queue_is_empty()
    print("Data-structures tests passed")
