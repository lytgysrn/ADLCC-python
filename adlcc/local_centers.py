"""Locally deep points, self-centrality level B* and local centers (Section 3 and 4.1).

* For every point x_i and level beta_j, the locally deep point of the
  neighbourhood N_{x_i}^{beta_j} is the member with the highest ILD at that
  level (``highest_ld[i, j]``).  Its frequency is the number of (i, j) in which
  it is chosen.
* The self-centrality level set B(x) collects the levels at which x is the
  locally deep point of its own neighbourhood; B*(x) is the level in B(x) with
  the largest ILD.  As in the reference R implementation, the first
  ``min_levels`` levels are not counted (an ILD averaged over fewer than 20
  levels is not considered stable) and B(x) must contain at least two levels.
* x is a local center if, among all points z whose neighbourhood at level
  B*(x) contains x, at least half have x as their locally deep point
  (the R implementation keeps proportions >= 0.5).
"""
import numpy as np
from numba import njit, prange
from scipy.stats import rankdata


@njit(parallel=True)
def _highest_ld(ranking_mat, dm0_order, nbr_list):
    n = ranking_mat.shape[0]
    n_levels = nbr_list.shape[0]
    highest = np.zeros((n, n_levels), dtype=np.int64)
    for i in prange(n):
        for j in range(n_levels):
            k = int(nbr_list[j])
            best = dm0_order[0, i]
            best_r = ranking_mat[best, j]
            for t in range(1, k):
                p = dm0_order[t, i]
                r = ranking_mat[p, j]
                if r < best_r:
                    best_r = r
                    best = p
            highest[i, j] = best
    return highest


def self_centrality(ILD_mat, dm0_order, nbr_list, min_levels=20):
    """Locally deep points of every neighbourhood and the self-centrality level of every point.

    Returns a dict with
      highest_ld (n, b)  highest_ld[i, j] = locally deep point of N_{x_i}^{beta_j},
      est_size   (n,)    neighbourhood size at B*(x_i); 0 if x_i has no self-centrality level,
      ild_bstar  (n,)    ILD of x_i at B*(x_i).
    """
    n_points = ILD_mat.shape[0]
    nbr_list = np.asarray(nbr_list, dtype=int)
    ranking_mat = rankdata(-ILD_mat, axis=0, method="ordinal").astype(np.int64)
    highest_ld = _highest_ld(
        ranking_mat,
        np.ascontiguousarray(dm0_order, dtype=np.int64),
        np.ascontiguousarray(nbr_list, dtype=np.int64),
    )
    est_size = np.zeros(n_points, dtype=int)
    ild_bstar = np.zeros(n_points, dtype=float)
    for i in range(n_points):
        own = np.where(highest_ld[i] == i)[0]
        own = own[own >= min_levels]
        if len(own) >= 2:
            pos = own[np.argmax(ILD_mat[i, own])]
            est_size[i] = nbr_list[pos]
            ild_bstar[i] = ILD_mat[i, pos]
    return {"highest_ld": highest_ld, "est_size": est_size, "ild_bstar": ild_bstar}


def local_centers(est_size, highest_ld, nbr_list, dm0_order):
    """Local centers (stability condition, proportion >= 0.5) sorted by decreasing frequency.

    Returns a dict with
      save_lc    (T,)  indices of the local centers, most frequent first,
      save_size  (T,)  their neighbourhood sizes at B*,
      freq_table dict  frequency of every locally deep point, decreasing (the ordered set E).
    """
    est_size = np.asarray(est_size, dtype=int)
    nbr_list = np.asarray(nbr_list, dtype=int)
    unique, counts = np.unique(np.asarray(highest_ld).ravel(), return_counts=True)
    freq_table = dict(zip(unique.astype(int).tolist(), counts.astype(int).tolist()))
    freq_table = dict(sorted(freq_table.items(), key=lambda kv: kv[1], reverse=True))

    accepted = []
    for i in np.where(est_size != 0)[0]:
        i = int(i)
        s = int(est_size[i])
        pos_s = np.where(nbr_list == s)[0]
        if len(pos_s) == 0:
            continue
        pos_s = int(pos_s[0])
        cols_with_i = np.any(dm0_order[:s, :] == i, axis=0)
        chosen = highest_ld[cols_with_i, pos_s]
        if chosen.size and float(np.mean(chosen == i)) >= 0.5:
            accepted.append(i)
    accepted.sort(key=lambda x: (-freq_table.get(x, 0), x))
    save_lc = np.array(accepted, dtype=int)
    return {"save_lc": save_lc, "save_size": est_size[save_lc], "freq_table": freq_table}
