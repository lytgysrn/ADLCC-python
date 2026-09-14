"""Adaptive grouping of the local centers (Section 4.2, Algorithm "reach").

Everything is expressed with the relative reachability of one point set towards
another,

    rho_{A|B} = mean over x in A of  reach(x -> B) / intra-reachability(x in A),

and two numbers derived from it: the background level phi_A (reachability of A
towards everything outside A; clipped to [0.9, 0.99] when used as a threshold)
and the disruption omega_{A|B} (fraction of points of A whose intra-reachability
drops once B is joined).

Before anything is merged (Section 4.2.1):
  * isolation: if no center lies within half the largest cell of another center,
    the cells are the groups;
  * cells without a boundary: a cell whose raw phi >= 1 reaches the outside at
    least as well as itself; if it is also looser than every cell it is directly
    linked to, it is not a unit and takes part in no pass (its points follow the
    center they are most similar to at the end).

For two groups A, B with a GLS contact (Section 4.2.2):
  * absorption: A is absorbed by B when rho_{A|B} >= 1 and rho_{B|A} > phi_B
    measured without A (a side that could be absorbed by several groups goes
    to the one it reaches best);
  * bond: two self-contained groups bond when, for both sides,
        rho_{s|.} > Q_th(s) = phi_s + omega_s^2 (max(q, phi_s) - phi_s),   q = 0.975,
    and their GLS contact is not weaker than q times the weakest GLS link that
    either group already relies on (no new weakest link).

Every merge must be carried by a community-level contact (Section 4.2.3): on
the depth graph, at least one pair of cells across the two sides must share
more weight than the configuration model predicts (Newman modularity gain
dq > 0).  An enclosed absorption (omega_A = 0) is exempt.

The pass is repeated on the new groups until nothing changes; finally groups
whose mass is below half the median mass are reconsidered against the group
they are most similar to (Section 4.2.4).
"""
import numpy as np
from scipy import sparse

from .similarity import reachable_similarity, intra_reachability, centers_isolated

Q_SLACK = 0.975          # q: an undisrupted bond may fall 2.5% short of self-reachability
PHI_CLIP = (0.9, 0.99)   # clip of the background level used in thresholds


def _stats(sym_dm, all_reach, pts):
    """(intra-reachability per point, clipped phi, raw phi) of a point set."""
    pts = np.asarray(pts, int)
    if pts.size <= 1:
        return np.ones(pts.size), PHI_CLIP[1], 0.0
    mcs = intra_reachability(reachable_similarity(sym_dm[np.ix_(pts, pts)]))
    mask = np.ones(all_reach.shape[0], bool)
    mask[pts] = False
    if not mask.any():
        return mcs, PHI_CLIP[1], 0.0
    raw = float(np.mean(np.mean(all_reach[np.ix_(pts, mask)], axis=1) / np.maximum(mcs, 1e-12)))
    return mcs, float(min(max(raw, PHI_CLIP[0]), PHI_CLIP[1])), raw


def _pair(sym_dm, pa, pb, mcs_a, mcs_b):
    """Directional rho and omega: A towards B, B towards A."""
    ids = np.concatenate([pa, pb])
    after = reachable_similarity(sym_dm[np.ix_(ids, ids)])
    mcs = intra_reachability(after)
    na = len(pa)
    rho_a = float(np.mean(np.mean(after[:na, na:], axis=1) / np.maximum(mcs_a, 1e-12)))
    rho_b = float(np.mean(np.mean(after[na:, :na], axis=1) / np.maximum(mcs_b, 1e-12)))
    om_a = float(np.mean(mcs[:na] < mcs_a - 1e-10))
    om_b = float(np.mean(mcs[na:] < mcs_b - 1e-10))
    return rho_a, rho_b, om_a, om_b


def _components(n, edges):
    par = list(range(n))

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            par[ra] = rb
    roots = [find(i) for i in range(n)]
    ids = {r: k for k, r in enumerate(dict.fromkeys(roots))}
    return [ids[r] for r in roots]


def _merge(groups, edges):
    comp = _components(len(groups), edges)
    merged = {}
    for k, g in enumerate(groups):
        merged.setdefault(comp[k], []).extend(g)
    return list(merged.values())


def _bottleneck(cells, W):
    """Weakest link of the maximum spanning tree of W over ``cells`` (inf for a singleton)."""
    if len(cells) <= 1:
        return np.inf
    es = sorted(((W[i, j], i, j) for x, i in enumerate(cells) for j in cells[x + 1:] if W[i, j] > 0), reverse=True)
    par = {c: c for c in cells}

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    left, last = len(cells), np.nan
    for w, i, j in es:
        fi, fj = find(i), find(j)
        if fi != fj:
            par[fi] = fj
            left -= 1
            last = w
            if left == 1:
                break
    return last if left == 1 else np.nan


def contact_gain(sym_dm, dm0_order, save_lc, nbr, G):
    """Newman modularity gain of joining two cells on the depth graph.

    Every observation keeps its top-r depth neighbours (r = size of the cell of
    the center it is most similar to); edges are weighted by sym_dm and
    symmetrised.  For cells i, j with weight e_ij between them, strengths
    k_i, k_j and total weight m:  dq[i, j] = e_ij / m - k_i k_j / (2 m^2).
    -inf where the cells have no GLS contact.
    """
    n = sym_dm.shape[0]
    owner = np.argmax(sym_dm[:, save_lc], axis=1)
    r = np.array([len(nbr[o]) for o in owner], int)
    rows, cols = [], []
    for i in range(n):
        nb = dm0_order[1:r[i] + 1, i]
        rows.append(np.full(nb.size, i))
        cols.append(nb)
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    A = sparse.csr_matrix((sym_dm[rows, cols], (rows, cols)), shape=(n, n))
    A = A.maximum(A.T)
    k = np.asarray(A.sum(axis=1)).ravel()
    m = 0.5 * k.sum()
    T = len(save_lc)
    ks = np.array([k[c].sum() for c in nbr])
    dq = np.full((T, T), -np.inf)
    for i in range(T):
        Ai = A[nbr[i]]
        for j in range(i + 1, T):
            if G[i, j] > 0:
                e = Ai[:, nbr[j]].sum()
                dq[i, j] = dq[j, i] = e / m - ks[i] * ks[j] / (2 * m * m)
    return dq


def _pass(groups, pts, st, sym_dm, G, q, all_reach, dq, birth=None, log=None, tag="L"):
    """One agglomeration pass over ``groups``; returns the accepted (a, b) edges.

    ``birth`` (weakest GLS link each group relies on) is only given at group level.
    """
    K = len(groups)
    rho = np.zeros((K, K))
    pairs = {}
    for a in range(K):
        for b in range(a + 1, K):
            if G[np.ix_(groups[a], groups[b])].max() <= 0:
                continue
            ra, rb, oa, ob = _pair(sym_dm, pts[a], pts[b], st[a][0], st[b][0])
            rho[a, b], rho[b, a] = ra, rb
            pairs[(a, b)] = (ra, rb, oa, ob)

    def phi_without(w, s):
        """Raw background of w with the candidate s left out of the outside."""
        mask = np.ones(all_reach.shape[0], bool)
        mask[pts[w]] = False
        mask[pts[s]] = False
        if not mask.any():
            return st[w][2]
        return float(np.mean(np.mean(all_reach[np.ix_(pts[w], mask)], axis=1) / np.maximum(st[w][0], 1e-12)))

    # absorption candidates: s reaches w at least as well as itself, and w does
    # not find s worse than its own background (Q_th at zero disruption)
    valid = {}
    for (a, b) in pairs:
        for s, w in ((a, b), (b, a)):
            if rho[s, w] >= 1.0 and rho[w, s] > min(phi_without(w, s), PHI_CLIP[1]):
                valid.setdefault(s, []).append(w)
    chosen = {s: max(ws, key=lambda w: rho[s, w]) for s, ws in valid.items()}

    edges = []
    for (a, b), (ra, rb, oa, ob) in pairs.items():
        verdict = "rej"
        if max(ra, rb) >= 1.0:
            if chosen.get(a) == b or chosen.get(b) == a:
                verdict = "ABS"
            elif b in valid.get(a, ()) or a in valid.get(b, ()):
                verdict = "nbest"
            else:
                verdict = "bgd"
        else:
            pa, pb = st[a][1], st[b][1]
            qa = pa + oa ** 2 * (max(q, pa) - pa)
            qb = pb + ob ** 2 * (max(q, pb) - pb)
            if ra > qa and rb > qb:
                verdict = "acc"
                if birth is not None and G[np.ix_(groups[a], groups[b])].max() < q * min(birth[a], birth[b]):
                    verdict = "DIP"
        if verdict in ("ABS", "acc") and dq[np.ix_(groups[a], groups[b])].max() <= 0:
            enclosed = verdict == "ABS" and ((chosen.get(a) == b and oa == 0.0) or (chosen.get(b) == a and ob == 0.0))
            if not enclosed:
                verdict = "NULL"
        if verdict in ("ABS", "acc"):
            edges.append((a, b))
        if log:
            extra = f" dq={dq[np.ix_(groups[a], groups[b])].max():.4f}"
            if birth is not None:
                extra += f" contact={G[np.ix_(groups[a], groups[b])].max():.3f} birth={min(birth[a], birth[b]):.3f}"
            log(f"{tag} {groups[a]}-{groups[b]} |{len(pts[a])}|{len(pts[b])}| rho={ra:.3f}/{rb:.3f} "
                f"om={oa:.2f}/{ob:.2f} phi={st[a][2]:.3f}/{st[b][2]:.3f} {verdict}{extra}")
    return edges


def _non_units(G, cell, nbr):
    """Cells with raw phi >= 1 that are looser than every directly linked candidate unit;
    a candidate without direct links is a non-unit when it is smaller than the median cell."""
    n = len(cell)
    cand = [i for i in range(n) if cell[i][2] >= 1.0]
    if not cand:
        return set()
    R = reachable_similarity(G)
    direct = (np.abs(G - R) < 1e-12) & (G > 0)
    loose = np.array([float(np.mean(c[0])) for c in cell])
    size = np.array([len(x) for x in nbr])
    med = float(np.median(size))
    out = set()
    for i in cand:
        nb = [j for j in np.where(direct[i])[0] if j not in cand]
        if nb:
            if loose[nb].min() > loose[i]:
                out.add(i)
        elif size[i] < med:
            out.add(i)
    return set() if len(out) >= n else out


def _attach_fragments(groups, mass_of, pts_of, sym_dm, all_reach, G, dq, log=None):
    """Reconsideration of small groups (Section 4.2.4).

    A group whose mass is below half the median mass goes to the group it is
    most similar to (GLS weighted by center-level reachability) provided the
    union is not more connected to the outside than the looser of the two parts,
    phi(A u B) <= max(phi_A, phi_B), the union is not larger than the largest
    group, and the contact is a community-level one (dq > 0).
    """
    groups = [list(g) for g in groups]
    W = G * reachable_similarity(G)
    while len(groups) > 2:
        mass = np.array([mass_of(g) for g in groups], float)
        low = np.median(mass) / 2
        frag = [i for i in np.argsort(mass) if mass[i] < low]
        if log:
            log(f"FR masses={[int(m) for m in mass]} low={low:.0f} fragments={[groups[i] for i in frag]}")
        if not frag or len(groups) - len(frag) < 2:
            break
        phi = [_stats(sym_dm, all_reach, pts_of(g))[2] for g in groups]
        moved = False
        for a in frag:
            sims = np.array([W[np.ix_(groups[a], groups[b])].max() if b != a else -1 for b in range(len(groups))])
            b = int(np.argmax(sims))
            if sims[b] <= 0:
                continue
            verdict, pu, origin = "keep", np.nan, np.nan
            if mass[a] + mass[b] > mass.max():
                verdict = "big"
            elif dq[np.ix_(groups[a], groups[b])].max() <= 0:
                verdict = "NULL"
            else:
                pu = _stats(sym_dm, all_reach, np.concatenate([pts_of(groups[a]), pts_of(groups[b])]))[2]
                origin = max(phi[a], phi[b])
                if pu <= origin:
                    verdict = "ATT"
            if log:
                log(f"FR {groups[a]}->{groups[b]} |{int(mass[a])}|{int(mass[b])}| phi={phi[a]:.3f}/{phi[b]:.3f} "
                    f"union={pu:.3f} origin={origin:.3f} {verdict}")
            if verdict == "ATT":
                groups[b] = groups[b] + groups[a]
                groups[a] = None
                moved = True
                break
        groups = [g for g in groups if g is not None]
        if not moved:
            break
    return groups


def group_local_centers(save_lc, sym_sm, nbr_save, sym_dm, dm0_order, q=Q_SLACK, log=None):
    """Group the local centers.

    save_lc   (T,)   local centers;  sym_sm (T, T) GLS;  nbr_save: cells;
    sym_dm    (n, n) symmetrised depth similarity;  dm0_order (n, n) neighbour order.
    log: optional callable receiving one text line per pairwise decision.

    Returns a dict with
      group_list  list of arrays of local centers (one per group),
      temp_clus   list of arrays of observations provisionally attached to each group
                  (every observation follows the unit center it is most similar to).
    """
    save_lc = np.asarray(save_lc, int)
    nbr = [np.asarray(x, int) for x in nbr_save]
    dm0_order = np.asarray(dm0_order, int)
    n = len(save_lc)
    G = np.array(sym_sm, float, copy=True)
    np.fill_diagonal(G, 0.0)
    all_reach = reachable_similarity(sym_dm)

    if n > 2 and centers_isolated(dm0_order, nbr, save_lc):
        if log:
            log("local centers isolated: cells are the groups")
        return {"group_list": [np.array([c]) for c in save_lc], "temp_clus": list(nbr)}

    cell = [_stats(sym_dm, all_reach, nbr[i]) for i in range(n)]
    unit = [i for i in range(n) if i not in _non_units(G, cell, nbr)]
    if log:
        log(f"cells without a boundary: {[i for i in range(n) if i not in unit]}")
    dq = contact_gain(sym_dm, dm0_order, save_lc, nbr, G)

    groups = [[i] for i in unit]
    edges = _pass(groups, [nbr[i] for i in unit], [cell[i] for i in unit], sym_dm, G, q, all_reach, dq, log=log, tag="L0")
    groups = _merge(groups, edges)

    while len(groups) > 1:
        pts = [np.concatenate([nbr[i] for i in g]) for g in groups]
        st = [_stats(sym_dm, all_reach, p) if len(g) > 1 else cell[g[0]] for g, p in zip(groups, pts)]
        birth = [_bottleneck(g, G) for g in groups]
        edges = _pass(groups, pts, st, sym_dm, G, q, all_reach, dq, birth=birth, log=log, tag="L+")
        if not edges:
            break
        groups = _merge(groups, edges)

    unit_arr = np.array(unit, int)
    owner = unit_arr[np.argmax(sym_dm[:, save_lc[unit_arr]], axis=1)]

    def _pts(g):
        return np.where(np.isin(owner, g))[0]

    groups = _attach_fragments(groups, lambda g: _pts(g).size, _pts, sym_dm, all_reach, G, dq, log=log)

    group_list = [save_lc[np.array(g, int)] for g in groups]
    temp_clus = []
    for g in groups:
        pts = _pts(g)
        temp_clus.append(pts if pts.size else np.concatenate([nbr[i] for i in g]))
    return {"group_list": group_list, "temp_clus": temp_clus}
