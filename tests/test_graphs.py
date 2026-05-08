from algorithms.cpu.graphs.bfs import bfs
from algorithms.cpu.graphs.dfs import dfs
from algorithms.cpu.graphs.dijkstra import dijkstra

# Shared test graph:
# 0 -- 1 -- 2
# |         |
# 3 ------- 4
GRAPH = {
    0: [1, 3],
    1: [0, 2],
    2: [1, 4],
    3: [0, 4],
    4: [2, 3],
}

WEIGHTED = {
    0: [(1, 1), (3, 4)],
    1: [(0, 1), (2, 2), (3, 5)],
    2: [(1, 2), (4, 1)],
    3: [(0, 4), (1, 5), (4, 3)],
    4: [(2, 1), (3, 3)],
}


# ── BFS ──────────────────────────────────────────────────────────────────────

def test_bfs_visits_all_nodes():
    order = bfs(GRAPH, 0)
    assert sorted(order) == [0, 1, 2, 3, 4]


def test_bfs_level_order():
    # From 0: immediate neighbours are 1 and 3 before deeper nodes
    order = bfs(GRAPH, 0)
    assert order[0] == 0
    assert set(order[1:3]) == {1, 3}


def test_bfs_single_node():
    assert bfs({0: []}, 0) == [0]


def test_bfs_disconnected_component():
    g = {0: [1], 1: [0], 2: [3], 3: [2]}
    order = bfs(g, 0)
    # Only the component containing 0 is reachable
    assert sorted(order) == [0, 1]


# ── DFS ──────────────────────────────────────────────────────────────────────

def test_dfs_visits_all_nodes():
    order = dfs(GRAPH, 0)
    assert sorted(order) == [0, 1, 2, 3, 4]


def test_dfs_starts_at_source():
    order = dfs(GRAPH, 0)
    assert order[0] == 0


def test_dfs_single_node():
    assert dfs({0: []}, 0) == [0]


def test_dfs_linear_graph():
    g = {0: [1], 1: [2], 2: [3], 3: []}
    order = dfs(g, 0)
    assert order == [0, 1, 2, 3]


# ── Dijkstra ─────────────────────────────────────────────────────────────────

def test_dijkstra_shortest_distances():
    dist, _ = dijkstra(WEIGHTED, 0)
    # 0→1 = 1, 0→2 = 3 (via 1), 0→3 = 4, 0→4 = 4 (via 1→2→4)
    assert dist[0] == 0
    assert dist[1] == 1
    assert dist[2] == 3
    assert dist[3] == 4
    assert dist[4] == 4


def test_dijkstra_prev_chain():
    dist, prev = dijkstra(WEIGHTED, 0)
    # Reconstruct path 0 → 4
    path = []
    node = 4
    while node in prev:
        path.append(node)
        node = prev[node]
    path.append(0)
    path.reverse()
    assert path[0] == 0
    assert path[-1] == 4
    # Total cost along recovered path should equal dist[4]
    total = sum(
        next(w for n, w in WEIGHTED[path[i]] if n == path[i + 1])
        for i in range(len(path) - 1)
    )
    assert total == dist[4]


def test_dijkstra_single_node():
    dist, prev = dijkstra({0: []}, 0)
    assert dist == {0: 0}
    assert prev == {}


def test_dijkstra_unreachable_node():
    g = {0: [(1, 2)], 1: [], 2: []}
    dist, _ = dijkstra(g, 0)
    assert dist[0] == 0
    assert dist[1] == 2
    assert 2 not in dist  # node 2 is unreachable


if __name__ == "__main__":
    test_bfs_visits_all_nodes()
    test_bfs_level_order()
    test_bfs_single_node()
    test_bfs_disconnected_component()
    test_dfs_visits_all_nodes()
    test_dfs_starts_at_source()
    test_dfs_single_node()
    test_dfs_linear_graph()
    test_dijkstra_shortest_distances()
    test_dijkstra_prev_chain()
    test_dijkstra_single_node()
    test_dijkstra_unreachable_node()
    print("Graph tests passed")
