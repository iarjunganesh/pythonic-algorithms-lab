"""Level-synchronous BFS frontier expansion with GPU acceleration.

When CuPy is available the frontier is represented as a boolean mask on GPU
and expanded via sparse matrix–vector multiply (SpMV) against the CSR
adjacency matrix, which is the standard GPU-BFS primitive.

Falls back to a CPU queue-based BFS when CuPy is not available.
"""
import numpy as np

try:
    import cupy as cp
    import cupyx  # noqa: F401 — ensures cupyx.scipy.sparse is importable
    from cupyx.scipy.sparse import csr_matrix as _cp_csr
except Exception:
    cp = None
    _cp_csr = None


# ── public helpers ────────────────────────────────────────────────────────────

def adj_to_csr(n_nodes, adj_dict):
    """Convert an adjacency dict ``{node: [neighbours]}`` to CSR arrays.

    Parameters
    ----------
    n_nodes : int
    adj_dict : dict

    Returns
    -------
    row_ptr : numpy.ndarray, shape (n_nodes + 1,), dtype int32
    col_idx : numpy.ndarray, shape (nnz,), dtype int32
    """
    row_ptr = np.zeros(n_nodes + 1, dtype=np.int32)
    for u in range(n_nodes):
        row_ptr[u + 1] = row_ptr[u] + len(adj_dict.get(u, []))
    nnz = int(row_ptr[n_nodes])
    col_idx = np.zeros(nnz, dtype=np.int32)
    for u in range(n_nodes):
        start = int(row_ptr[u])
        for k, v in enumerate(adj_dict.get(u, [])):
            col_idx[start + k] = int(v)
    return row_ptr, col_idx


def bfs_frontier(n_nodes, row_ptr, col_idx, source=0):
    """Level-synchronous BFS returning per-node distances from *source*.

    Parameters
    ----------
    n_nodes : int
        Number of nodes.
    row_ptr : array-like, int32, length n_nodes + 1
        CSR row-pointer array.
    col_idx : array-like, int32, length nnz
        CSR column-index array.
    source : int
        Source node.

    Returns
    -------
    numpy.ndarray, shape (n_nodes,), dtype int32
        BFS level (distance) for each node; -1 if unreachable.
    """
    row_ptr = np.asarray(row_ptr, dtype=np.int32)
    col_idx = np.asarray(col_idx, dtype=np.int32)
    if cp is not None and _cp_csr is not None:
        return _bfs_cupy(n_nodes, row_ptr, col_idx, source)
    return _bfs_cpu(n_nodes, row_ptr, col_idx, source)


# ── GPU path (CuPy SpMV) ──────────────────────────────────────────────────────

def _bfs_cupy(n_nodes, row_ptr, col_idx, source):
    nnz = len(col_idx)
    data = np.ones(nnz, dtype=np.float32)
    # Build CSR on GPU (transpose so col-neighbours map to row indices)
    A = _cp_csr(
        (cp.asarray(data), cp.asarray(col_idx), cp.asarray(row_ptr)),
        shape=(n_nodes, n_nodes),
    )
    A_T = A.T.tocsr()

    levels = cp.full(n_nodes, -1, dtype=cp.int32)
    frontier = cp.zeros(n_nodes, dtype=cp.float32)
    visited = cp.zeros(n_nodes, dtype=cp.bool_)

    levels[source] = 0
    frontier[source] = 1.0
    visited[source] = True
    level = 0

    while bool(cp.any(frontier > 0)):
        next_raw = A_T.dot(frontier)          # propagate frontier to neighbours
        new_mask = (next_raw > 0) & (~visited)
        if not bool(cp.any(new_mask)):
            break
        level += 1
        levels[cp.where(new_mask)[0]] = level
        visited |= new_mask
        frontier = new_mask.astype(cp.float32)

    return cp.asnumpy(levels)


# ── CPU fallback (queue-based) ────────────────────────────────────────────────

def _bfs_cpu(n_nodes, row_ptr, col_idx, source):
    from collections import deque

    levels = np.full(n_nodes, -1, dtype=np.int32)
    levels[source] = 0
    q = deque([source])
    while q:
        u = q.popleft()
        for idx in range(int(row_ptr[u]), int(row_ptr[u + 1])):
            v = int(col_idx[idx])
            if levels[v] == -1:
                levels[v] = levels[u] + 1
                q.append(v)
    return levels
