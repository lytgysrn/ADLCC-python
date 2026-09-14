"""Cells, group-level local similarity (GLS) and reachable similarity (Section 4.1-4.2).

* ``symmetrize``: the symmetrised depth-based similarity S.
* ``group_similarity``: every observation joins the cell G_t of the local
  center it is most similar to; cells of size one are dissolved.  For two
  cells that are in contact (a center is at least as similar to a point of the
  other cell as the least similar pair inside its own cell) the GLS is the
  rank-based similarity of Section 4.1, normalised by the cell sizes.
* ``reachable_similarity``: bottleneck (maximin path) similarity on a weighted
  graph; ``intra_reachability`` is its per-point mean towards the other members.
* ``centers_isolated``: the paper's ``ifmerging`` check; True when no center
  lies within half the largest cell of another center.
"""
import numpy as np
from numba import njit


def symmetrize(dm):
    return (dm + dm.T) / 2


def _cells(sym_dm, save_lc):
    assignments = save_lc[np.argmax(sym_dm[:, save_lc], axis=1)]
    counts = np.array([np.sum(assignments == c) for c in save_lc])
    return assignments, counts


def group_similarity(save_lc, sym_dm):
    """Cells and GLS matrix of the local centers.

    Returns a dict with
      save_lc  (T,)     local centers kept (cells of one point are dissolved),
      nbr_save list     nbr_save[t] = indices of the points in cell G_t,
      counts   (T,)     cell sizes,
      TFmatrix (T, T)   cells in contact,
      sym_sm   (T, T)   symmetric GLS matrix.
    """
    save_lc = np.asarray(save_lc, dtype=int)
    if len(save_lc) == 0:
        return {"save_lc": save_lc, "nbr_save": [], "counts": np.array([], int),
                "TFmatrix": np.zeros((0, 0), bool), "sym_sm": np.zeros((0, 0))}
    assignments, counts = _cells(sym_dm, save_lc)
    while np.any(counts == 1) and len(save_lc) > 1:
        save_lc = save_lc[counts != 1]
        assignments, counts = _cells(sym_dm, save_lc)
    T = len(save_lc)
    nbr_save = [np.where(assignments == c)[0] for c in save_lc]

    TF = np.zeros((T, T), dtype=bool)
    for g in range(T):
        pts = nbr_save[g]
        inter_min = np.min(sym_dm[np.ix_(pts, pts)])
        for h in range(T):
            if h != g:
                TF[g, h] = np.max(sym_dm[save_lc[g], nbr_save[h]]) >= inter_min
    TF = TF | TF.T
    sym_sm = _gls(sym_dm, assignments, TF, nbr_save, counts)
    return {"save_lc": save_lc, "nbr_save": nbr_save, "counts": counts, "TFmatrix": TF, "sym_sm": sym_sm}


def _gls(d, assignments, TF, nbr_save, counts):
    """Group-level local similarity between cells in contact (definition of Section 4.1)."""
    T = len(nbr_save)
    c = np.zeros((T, T))
    for gi in range(T):
        for gj in range(gi + 1, T):
            if not TF[gi, gj]:
                continue
            Gi, Gj = nbr_save[gi], nbr_save[gj]
            idx_i = np.where(assignments == assignments[Gi[0]])[0]
            idx_j = np.where(assignments == assignments[Gj[0]])[0]
            d_Gj = d[Gj]
            for x in Gi:
                dx = d[x]
                dxy = dx[Gj]
                dyx = d_Gj[:, x]
                mask = (dx[None, :] >= dxy[:, None]) | (d_Gj >= dyx[:, None])
                my, mx = mask[:, idx_j], mask[:, idx_i]
                ny, nx = my.sum(axis=1), mx.sum(axis=1)
                wy = 1.0 * (dx[idx_j] > d_Gj[:, idx_j]) + 0.5 * (dx[idx_j] == d_Gj[:, idx_j])
                wx = 1.0 * (dx[idx_i] > d_Gj[:, idx_i]) + 0.5 * (dx[idx_i] == d_Gj[:, idx_i])
                vy, vx = ny > 0, nx > 0
                if np.any(vy):
                    c[gi, gj] += np.sum((wy * my)[vy].sum(axis=1) / ny[vy])
                if np.any(vx):
                    c[gj, gi] += np.sum(((1.0 - wx) * mx)[vx].sum(axis=1) / nx[vx])
    normalized = np.zeros_like(c)
    for i in range(T):
        normalized[i, :] = c[i, :] / (counts * counts[i])
    return symmetrize(normalized)


@njit
def _reachable(sim):
    n = sim.shape[0]
    R = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        R[i, i] = 1.0
    n_edges = n * (n - 1) // 2
    us = np.empty(n_edges, dtype=np.int32)
    vs = np.empty(n_edges, dtype=np.int32)
    ws = np.empty(n_edges, dtype=np.float64)
    e = 0
    for i in range(n):
        for j in range(i + 1, n):
            w = sim[i, j]
            if w > 0.0:
                us[e] = i
                vs[e] = j
                ws[e] = w
                e += 1
    if e == 0:
        return R
    order = np.argsort(-ws[:e])
    parent = np.arange(n)
    head = np.arange(n)
    nxt = -np.ones(n, dtype=np.int64)
    size = np.ones(n, dtype=np.int64)
    for k in range(e):
        t = order[k]
        u = int(us[t])
        v = int(vs[t])
        w = ws[t]
        cu = u
        while parent[cu] != cu:
            parent[cu] = parent[parent[cu]]
            cu = parent[cu]
        cv = v
        while parent[cv] != cv:
            parent[cv] = parent[parent[cv]]
            cv = parent[cv]
        if cu == cv:
            continue
        if size[cu] < size[cv]:
            cu, cv = cv, cu
        i = head[cu]
        while i >= 0:
            j = head[cv]
            while j >= 0:
                R[i, j] = w
                R[j, i] = w
                j = nxt[j]
            i = nxt[i]
        i = head[cu]
        while nxt[i] >= 0:
            i = nxt[i]
        nxt[i] = head[cv]
        size[cu] += size[cv]
        parent[cv] = cu
    return R


def reachable_similarity(sim):
    """Reachable similarity: for every pair, the largest bottleneck over all paths (Kruskal order)."""
    n = sim.shape[0]
    if n <= 1:
        return np.ones((n, n)) if n == 1 else np.zeros((0, 0))
    if n == 2:
        return np.asarray(sim, dtype=float).copy()
    return _reachable(np.ascontiguousarray(sim, dtype=np.float64))


def intra_reachability(reach):
    """Mean reachable similarity of every member towards the other members (diagonal excluded)."""
    return (np.sum(reach, axis=0) - 1) / (reach.shape[0] - 1)


def centers_isolated(dm0_order, nbr_save, save_lc):
    """True when no local center is among the first (max cell size / 2) neighbours of another one."""
    if len(save_lc) == 2:
        return True
    size = round(max(len(nbr) for nbr in nbr_save) / 2)
    others = set(int(c) for c in save_lc)
    for c in save_lc:
        nbrs = dm0_order[1:size, c] if size > 1 else np.array([], int)
        if np.any(np.isin(nbrs, list(others - {int(c)}))):
            return False
    return True
