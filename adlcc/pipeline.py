"""End-to-end A-DLCC.

    depth  = depth_stage(X)                  # depth-based similarity, ILD (expensive, cache it)
    result = adlcc(X, depth, classifier="rf")
    result["labels"]                         # cluster label of every observation, 1..K

The only inputs are the data and the choice of the final classifier; there is
no tuning parameter.
"""
import numpy as np

from .depth import depth_similarity
from .local_depth import integrated_local_depth
from .local_centers import self_centrality, local_centers
from .similarity import symmetrize, group_similarity
from .grouping import group_local_centers
from .temp_clusters import temporary_clusters
from .classify import assign_remaining, clusters_to_labels


def depth_stage(data, device=None):
    """Depth-based similarity matrix and integrated local depth of ``data``.

    Returns a dict with dm (n, n), Lmatrix (n, n), ld_mat, ILD_mat (n, b), dm0_order (n, n), nbr_list (b,).
    device: see ``adlcc.depth.depth_similarity``.
    """
    data = np.asarray(data, dtype=float)
    out = depth_similarity(data, device=device)
    ild = integrated_local_depth(data, out["dm"], out["Lmatrix"])
    return {"dm": out["dm"], "Lmatrix": out["Lmatrix"], **ild}


def adlcc(data, depth=None, classifier="rf", n_trees=100, k_knn=7, device=None, log=None):
    """Run A-DLCC on ``data``.

    depth       output of ``depth_stage`` (computed here if None),
    classifier  "rf" (random forest), "knn" (depth-based kNN) or "maxdep" for the final step,
    log         optional callable receiving one line per grouping decision.

    Returns a dict with
      local_centers  indices of the local centers (frequency order),
      cells          points attached to each local center,
      groups         groups of local centers,
      temp_clus      temporary clusters (lists of point indices),
      temp_labels    label vector of the temporary clustering (0 = not yet assigned),
      labels         final label vector, 1..K,
      n_clusters     K.
    """
    data = np.asarray(data, dtype=float)
    n = data.shape[0]
    if depth is None:
        depth = depth_stage(data, device=device)
    dm, dm0_order, nbr_list = depth["dm"], depth["dm0_order"], depth["nbr_list"]
    sym_dm = symmetrize(dm)

    sc = self_centrality(depth["ILD_mat"], dm0_order, nbr_list)
    lc = local_centers(sc["est_size"], sc["highest_ld"], nbr_list, dm0_order)
    gs = group_similarity(lc["save_lc"], sym_dm)
    gr = group_local_centers(gs["save_lc"], gs["sym_sm"], gs["nbr_save"], sym_dm, dm0_order, log=log)
    tc = temporary_clusters(sym_dm, gr["group_list"], gr["temp_clus"], data, lc["freq_table"])
    labels = assign_remaining(data, tc["temp_clus"], method=classifier, dm=dm, n_trees=n_trees, k_knn=k_knn)
    return {
        "local_centers": gs["save_lc"],
        "cells": gs["nbr_save"],
        "groups": tc["group_list"],
        "temp_clus": tc["temp_clus"],
        "temp_labels": clusters_to_labels(n, tc["temp_clus"]),
        "labels": labels,
        "n_clusters": len(tc["temp_clus"]),
    }
