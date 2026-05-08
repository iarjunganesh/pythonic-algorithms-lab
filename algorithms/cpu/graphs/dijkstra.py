import heapq


def dijkstra(adj, src):
    """Compute shortest paths from `src`.

    `adj` is a dict mapping node -> list of (neighbor, weight).
    Returns (dist, prev) where dist[node] is shortest distance.
    """
    dist = {src: 0}
    prev = {}
    pq = [(0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, float('inf')):
            continue
        for v, w in adj.get(u, []):
            nd = d + w
            if nd < dist.get(v, float('inf')):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    return dist, prev
