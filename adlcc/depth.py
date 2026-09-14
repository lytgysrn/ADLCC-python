"""Spatial depth and the depth-based similarity matrix.

``spatial_depth`` is the spatial depth of query points with respect to a
reference sample.  ``depth_similarity`` builds the n x n matrix

    S[j, i] = D(x_i | X_{R x_j}),

the spatial depth of x_i in the sample reflected about x_j (the depth-based
similarity of DLCC).  Both a NumPy and a PyTorch (GPU) implementation are
provided; they compute the same quantity in double precision.
"""
import numpy as np

_TOL = 1e-5


def spatial_depth(x, data):
    """Spatial depth of each row of ``x`` with respect to the sample ``data``."""
    data = np.asarray(data, dtype=float)
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    n = data.shape[0]
    dep = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        a = x[i][None, :] - data
        a = a[np.sum(np.abs(a), axis=1) != 0]
        a = a / np.sqrt(np.sum(a ** 2, axis=1))[:, None]
        e = np.sum(a, axis=0) / n
        dep[i] = 1 - np.sqrt(np.sum(e ** 2))
    return dep


def depth_similarity_numpy(data):
    """Depth-based similarity matrix with NumPy (O(n^3 d); fine for n up to ~1000)."""
    data = np.asarray(data, dtype=float)
    n, d = data.shape
    rn_inv = 1.0 / (2 * n - 1)
    dm = np.zeros((n, n))
    Ematrix = np.zeros((n, d))
    Lmatrix = np.zeros((n, n))
    for i in range(n):
        a = -data + data[i]
        Lmatrix[:, i] = np.linalg.norm(a, axis=1)
        norm_a = Lmatrix[:, i].copy()
        norm_a[norm_a < _TOL] = 1
        Ematrix[i] = np.sum(a / norm_a[:, None], axis=0)
    Lmatrix_save = Lmatrix.copy()
    Lmatrix = Lmatrix ** 2
    two_L = 2 * Lmatrix
    for j in range(n):
        idx = np.arange(n) != j
        b_temp = data[idx] - 2 * data[j]
        norm_bM = np.sqrt(np.abs(two_L[idx, j][:, None] + two_L[j, :] - Lmatrix[idx, :]))
        norm_bM[norm_bM < _TOL] = np.inf
        norm_bM = 1 / norm_bM
        C = (b_temp.T @ norm_bM).T
        C = C + np.sum(norm_bM, axis=0)[:, None] * data + Ematrix
        dm[j] = 1 - np.linalg.norm(C * rn_inv, axis=1)
    return {"dm": dm, "Lmatrix": Lmatrix_save}


def depth_similarity_torch(data, device="cuda"):
    """Depth-based similarity matrix with PyTorch (same algorithm, float64)."""
    import torch

    data = np.asarray(data, dtype=float)
    n, d = data.shape
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    X = torch.as_tensor(data, dtype=torch.float64, device=device)
    rn_inv = 1.0 / (2 * n - 1)
    dm = torch.zeros((n, n), dtype=torch.float64, device=device)
    Ematrix = torch.zeros((n, d), dtype=torch.float64, device=device)
    Lmatrix = torch.zeros((n, n), dtype=torch.float64, device=device)
    for i in range(n):
        a = -X + X[i]
        Lmatrix[:, i] = torch.norm(a, dim=1)
        norm_a = Lmatrix[:, i].clone()
        norm_a[norm_a < _TOL] = 1
        Ematrix[i] = (a / norm_a[:, None]).sum(dim=0)
    Lmatrix_save = Lmatrix.clone()
    Lmatrix = Lmatrix ** 2
    for j in range(n):
        idx = torch.arange(n, device=device) != j
        b_temp = X[idx] - 2 * X[j]
        b1 = (2 * Lmatrix[idx, j]).unsqueeze(1)
        b2 = (2 * Lmatrix[j, :]).unsqueeze(0)
        norm_bM = torch.sqrt(torch.abs(b1 + b2 - Lmatrix[idx, :]))
        norm_bM[norm_bM < _TOL] = float("inf")
        norm_bM = 1 / norm_bM
        C = (b_temp.T @ norm_bM).T
        C = C + norm_bM.sum(dim=0).unsqueeze(1) * X + Ematrix
        dm[j] = 1 - torch.norm(C * rn_inv, dim=1)
    return {"dm": dm.cpu().numpy(), "Lmatrix": Lmatrix_save.cpu().numpy()}


GPU_MIN_N = 800   # default device rule used for all results in the paper


def depth_similarity(data, device=None):
    """Depth-based similarity matrix ``dm`` and the Euclidean distance matrix ``Lmatrix``.

    device: None   -> NumPy for n < 800, otherwise PyTorch on the GPU when CUDA is
                      available (this is the rule under which the paper's results were computed);
            "numpy"        -> NumPy;
            "cuda" / "cpu" -> PyTorch on that device.
    The two implementations agree to ~1e-15; the difference can only matter when
    the data contain exact ties (duplicate observations), through the neighbour order.
    """
    if device == "numpy":
        return depth_similarity_numpy(data)
    if device is None:
        if np.asarray(data).shape[0] < GPU_MIN_N:
            return depth_similarity_numpy(data)
        try:
            import torch
            if not torch.cuda.is_available():
                return depth_similarity_numpy(data)
        except ImportError:
            return depth_similarity_numpy(data)
        device = "cuda"
    return depth_similarity_torch(data, device=device)
