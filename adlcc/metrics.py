"""External clustering metrics: ARI, NMI and purity."""
import numpy as np
from scipy.stats import mode
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def cluster_performance(y_true, y_pred):
    """ARI, NMI and purity of ``y_pred`` against ``y_true`` (both label vectors)."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    correct = 0
    for c in np.unique(y_pred):
        m = y_pred == c
        correct += np.sum(y_true[m] == mode(y_true[m], keepdims=True).mode[0])
    return {
        "ARI": float(adjusted_rand_score(y_true, y_pred)),
        "NMI": float(normalized_mutual_info_score(y_true, y_pred)),
        "Purity": float(correct / len(y_true)),
    }
