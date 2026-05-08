from collections import deque


def bfs(graph, start):
    """Breadth-first traversal. `graph` is adjacency mapping."""
    q = deque([start])
    visited = {start}
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in graph.get(u, []):
            if v not in visited:
                visited.add(v)
                q.append(v)
    return order
