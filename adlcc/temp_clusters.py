"""Temporary clusters (Section 4.3).

Given the groups of local centers, every observation is attached to the local
center it is most similar to.  Points whose closest similarity lies outside
their own group are set aside; a representativeness score (difference between
the similarity to the own group and to the other groups) then decides which
points remain in the temporary clusters and which are left to the final
classification step.  Groups consisting of a single local center are first
completed with the most representative unused locally deep point.
"""
import numpy as np

from .depth import spatial_depth


def _check_point(group_list, sym_dm, assignments, n):
    """Initial temporary clusters and set-aside points."""
    K = len(group_list)
    temp_clus = [[] for _ in range(K)]
    left_clus = [[] for _ in range(K)]
    for i in range(K):
        a_c = group_list[i]
        if len(a_c) > 1:
            nbr_ac = [np.where(assignments == x)[0] for x in a_c]
            ng_points = np.setdiff1d(np.arange(n), np.concatenate(nbr_ac))
            for j in range(len(a_c)):
                points = nbr_ac[j]
                other_points = np.concatenate([nbr_ac[k] for k in range(len(a_c)) if k != j])
                is_vals = np.max(sym_dm[np.ix_(points, other_points)], axis=1)
                if len(ng_points) == 0:
                    keep = np.ones(len(points), dtype=bool)
                else:
                    bs_vals = np.max(sym_dm[np.ix_(points, ng_points)], axis=1)
                    keep = is_vals - bs_vals > 0
                keep[points == a_c[j]] = True
                temp_clus[i].extend(points[keep])
                left_clus[i].extend(points[~keep])
        else:
            points = np.where(assignments == a_c[0])[0]
            ng_points = np.setdiff1d(np.arange(n), points)
            if len(ng_points) == 0:
                temp_clus[i] = points.tolist()
                continue
            overlap = np.array([np.sum(sym_dm[j, ng_points] >= sym_dm[j, a_c[0]]) for j in points])
            keep = overlap == 0
            temp_clus[i] = points[keep].tolist()
            left_clus[i] = points[~keep].tolist()
    return temp_clus, left_clus


def _assign_score(Kclus, sym_dm, clusters, group_list):
    """Representativeness score of every point of ``clusters`` towards its own group."""
    a = np.array(np.concatenate(group_list), dtype=int)
    scores = [[] for _ in range(Kclus)]
    for k in range(Kclus):
        if len(clusters[k]) == 0:
            continue
        own = np.array(group_list[k], dtype=int)
        other = np.setdiff1d(a, group_list[k])
        for x in clusters[k]:
            idx = int(x)
            ma = np.max(sym_dm[own, idx])
            mb = np.max(sym_dm[other, idx]) if len(other) else 0.0
            scores[k].append((ma - mb) / (ma if ma - mb > 0 else mb))
    return scores


def temporary_clusters(sym_dm, group_list, provisional, data, freq_table):
    """Temporary clusters for the groups of local centers.

    sym_dm       (n, n) symmetrised depth similarity,
    group_list   groups of local centers (from ``group_local_centers``),
    provisional  provisional members of each group (used to complete singleton groups),
    data         (n, d) observations,
    freq_table   frequency-ordered locally deep points (from ``local_centers``).

    Returns a dict with
      temp_clus   list of lists of point indices (one temporary cluster per group),
      group_list  groups, with singleton groups completed.
    """
    group_list = [np.asarray(g) for g in group_list]
    if not group_list:
        return {"temp_clus": [], "group_list": []}
    n = sym_dm.shape[0]
    Kclus = len(group_list)
    a = np.concatenate(group_list)

    singles = [i for i, g in enumerate(group_list) if len(g) == 1]
    if singles:
        candidates = np.array([x for x in freq_table if x not in a], dtype=int)
        for v in singles:
            n_can = len(candidates)
            if n_can == 0:
                break
            others = np.setdiff1d(a, group_list[v])
            if len(others) == 0:
                continue
            to_own = sym_dm[candidates, group_list[v]]
            to_others = np.max(sym_dm[np.ix_(candidates, others)], axis=1)
            ranks_idx = np.where(to_own > to_others)[0]
            if len(ranks_idx) == 0:
                continue
            cand_v = candidates[ranks_idx]
            depth_value = spatial_depth(data[cand_v], data[np.asarray(provisional[v], int)]) * (n_can - ranks_idx) / n_can
            best = cand_v[np.argmax(depth_value)]
            group_list[v] = np.append(group_list[v], best)
            candidates = candidates[candidates != best]
        a = np.concatenate(group_list)

    a_int = np.array(a, dtype=int)
    assignments = a_int[np.argmax(sym_dm[:, a_int], axis=1)]
    temp_clus, left_clus = _check_point(group_list, sym_dm, assignments, n)

    score_temp = _assign_score(Kclus, sym_dm, temp_clus, group_list)
    if sum(len(lc) for lc in left_clus) > 0:
        score_left = _assign_score(Kclus, sym_dm, left_clus, group_list)
        flat = [s for sub in score_left for s in sub]
        mean_left = np.mean(flat) if flat else 0.0
        lq = [mean_left if len(score_left[x]) <= len(score_temp[x]) else 0.5 * mean_left for x in range(Kclus)]

        # points of the temporary clusters scoring below the set-aside level are set aside too
        for i in range(Kclus):
            index = [j for j, s in enumerate(score_temp[i]) if s < lq[i]]
            if index:
                left_clus[i].extend([temp_clus[i][j] for j in index])
                score_left[i].extend([score_temp[i][j] for j in index])
                temp_clus[i] = [temp_clus[i][j] for j in range(len(temp_clus[i])) if j not in index]
                score_temp[i] = [score_temp[i][j] for j in range(len(score_temp[i])) if j not in index]

        # border score per cluster: largest gap in the lower half of the kept scores,
        # or the quantile that brings the cluster to half of its points
        new_border = []
        for i in range(Kclus):
            n_cur = len(temp_clus[i])
            n_tot = n_cur + len(left_clus[i])
            cdd_1 = 1
            if len(score_temp[i]) > 0:
                sortscore = np.sort(score_temp[i])[::-1]
                start, end = max(0, n_cur // 2 - 1), min(n_cur, len(sortscore))
                if end > start:
                    sub = sortscore[start:end]
                    cdd_1 = sub[np.argmin(np.diff(sub))] if len(sub) > 1 else sub[0]
            cdd_2 = 1
            if n_cur < 0.5 * n_tot:
                temp_s = 1 - (0.5 * n_tot - n_cur) / len(score_left[i])
                cdd_2 = np.quantile(score_left[i], temp_s) if temp_s > 0 and len(score_left[i]) > 0 else 0
            new_border.append(min(cdd_1, cdd_2))

        for x in range(Kclus):
            if len(score_left[x]) > 0:
                left_clus[x] = [left_clus[x][j] for j, s in enumerate(score_left[x]) if s > new_border[x]]
        for i in range(Kclus):
            temp_clus[i].extend(left_clus[i])

    return {"temp_clus": temp_clus, "group_list": group_list}
