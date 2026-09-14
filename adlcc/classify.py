"""Final step of the DLCC framework: assign the remaining observations to a cluster.

Three classifiers are supported, as in DLCC: depth-based kNN on the
depth-based similarity (``"knn"``), maximum spatial depth (``"maxdep"``) and a
random forest (``"rf"``, 100 trees, minimum leaf size 3, i.e. the defaults of
the R ``randomForest`` package with ``nodesize = 3``).  The random forest uses
NumPy's global random state; seed it with ``numpy.random.seed`` for
reproducible runs.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .depth import spatial_depth


def clusters_to_labels(n, clusters):
    """Label vector: 1..K for the members of ``clusters``, 0 for unassigned points."""
    labels = np.zeros(n, dtype=int)
    for i, clus in enumerate(clusters):
        labels[np.asarray(clus, dtype=int)] = i + 1
    return labels


def depth_knn(sim, classes, k, K):
    """Depth-based kNN: sim (n_query, n_labelled), classes in 1..k; K neighbours; ties by best similarity."""
    n_query = sim.shape[0]
    counts = np.zeros((n_query, k), dtype=int)
    for j in range(n_query):
        nearest = classes[np.argsort(sim[j, :])[::-1][:K]]
        for i in range(k):
            counts[j, i] = np.sum(nearest == i + 1)
    out = np.zeros(n_query, dtype=int)
    for x in range(n_query):
        best = np.where(counts[x, :] == np.max(counts[x, :]))[0] + 1
        if len(best) > 1:
            max_sim = [np.max(sim[x, classes == c]) if np.any(classes == c) else 0 for c in best]
            best = best[int(np.argmax(max_sim))]
        else:
            best = best[0]
        out[x] = best
    return out


def assign_remaining(data, temp_clus, method="rf", dm=None, n_trees=100, k_knn=7):
    """Classify the observations outside the temporary clusters.

    Returns the full label vector (1..K).  ``dm`` (the depth-based similarity
    matrix, as computed, not symmetrised) is required for ``method="knn"``.
    """
    X = np.asarray(data)
    n = X.shape[0]
    K = len(temp_clus)
    labels = clusters_to_labels(n, temp_clus)
    left = np.where(labels == 0)[0]
    if len(left) == 0:
        return labels
    if method == "rf":
        rf = RandomForestClassifier(n_estimators=n_trees, min_samples_leaf=3, bootstrap=True)
        rf.fit(X[labels != 0], labels[labels != 0])
        labels[left] = rf.predict(X[left])
    elif method == "knn":
        if dm is None:
            raise ValueError("dm is required for method='knn'")
        labelled = np.where(labels != 0)[0]
        labels[left] = depth_knn(dm[np.ix_(left, labelled)], labels[labelled], K, k_knn)
    elif method == "maxdep":
        depth_mat = np.zeros((len(left), K))
        for j in range(K):
            depth_mat[:, j] = spatial_depth(X[left], X[np.asarray(temp_clus[j], int)])
        labels[left] = np.argsort(depth_mat, axis=1)[:, -1] + 1
    else:
        raise ValueError("method must be 'rf', 'knn' or 'maxdep'")
    return labels
