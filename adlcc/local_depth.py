"""beta-local depth and beta-integrated local depth (ILD) at every locality level.

For each point x_j and each neighbourhood size k in ``nbr_list`` the local depth
LD^beta(x_j) is the spatial depth of x_j with respect to its k most similar
points (Section 2 of the paper); the ILD is the running mean of LD over the
levels (Section 3, uniform weights).  The neighbourhood sizes run from
min(2d, 10) to n in steps of one, i.e. beta_{i+1} = beta_i + 1/n and beta_b = 1.
"""
import numpy as np

_TOL = 1e-5


def integrated_local_depth(data, dm, Lmatrix, nbr_list=None):
    """ILD matrix from the depth-based similarity ``dm`` and the distance matrix ``Lmatrix``.

    Returns a dict with
      ld_mat    (n, b)  local depth at every level,
      ILD_mat   (n, b)  integrated local depth (cumulative mean of ld_mat),
      dm0_order (n, n)  column j lists the points in decreasing similarity to x_j
                        (column j starts with j itself),
      nbr_list  (b,)    neighbourhood sizes.
    """
    data = np.asarray(data, dtype=float)
    n, d = data.shape
    if nbr_list is None:
        nbr_list = np.arange(min(2 * d, 10), n + 1)
    nbr_list = np.asarray(nbr_list, dtype=int)
    b = len(nbr_list)
    ld_mat = np.zeros((n, b))
    dm0_order = np.argsort(-dm, axis=1).T

    inv_L = np.array(Lmatrix, dtype=float, copy=True)
    inv_L[inv_L < _TOL] = np.inf
    inv_L = 1 / inv_L
    last = nbr_list - 1
    for j in range(n):
        label = dm0_order[: nbr_list.max(), j]
        w = inv_L[j, label]
        norm_csum = np.cumsum(w)[last]
        C2 = np.outer(norm_csum, data[j])
        C = np.cumsum(data[label] * w[:, None], axis=0)[last, :]
        mean_C = (C2 - C) / nbr_list[:, None]
        ld_mat[j, :] = 1 - np.sqrt(np.sum(mean_C ** 2, axis=1))
    ILD_mat = np.cumsum(ld_mat, axis=1) / (np.arange(b) + 1)
    return {"ld_mat": ld_mat, "ILD_mat": ILD_mat, "dm0_order": dm0_order, "nbr_list": nbr_list}
