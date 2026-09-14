"""Data sets of the paper (Section 5) and the classifier used for each.

Real data (Table 4 and the Anuran case study) and the four synthetic data sets
of Figure 5.  Iris, Wine and BC come with scikit-learn; Seed and Pa are the UCI
files (copies in ``data/``); the remaining files are in ``data/`` as well.
Preprocessing follows the paper: Wine, Pa, BC and Seg are standardised, the
others are used as they are.
"""
import os

import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.preprocessing import StandardScaler

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

# final classifier of Table 4 / Section 5.2.2: depth-based kNN (K = 7) for Seed, BC
# and Optidigits, random forest (100 trees) for the others
CLASSIFIER = {
    "Iris": "rf", "Seed": "knn", "Wine": "rf", "Pa": "rf", "BC": "knn", "Seg": "rf",
    "Yale-B": "rf", "Optidigits": "knn", "Anuran": "rf",
    "Starbeam": "rf", "Blend": "rf", "Bainba": "rf", "Agg": "knn",
}
REAL = ["Iris", "Seed", "Wine", "Pa", "BC", "Seg", "Yale-B", "Optidigits", "Anuran"]
SYNTHETIC = ["Starbeam", "Blend", "Bainba", "Agg"]
ALL = REAL + SYNTHETIC


def _mat(name, key, label_key):
    m = sio.loadmat(os.path.join(DATA, name))
    return m[key].astype(float), m[label_key].ravel()


def load_dataset(name):
    """Return (X, y) for a data set of the paper."""
    if name == "Iris":
        d = load_iris()
        return d.data.astype(float), d.target
    if name == "Seed":
        df = pd.read_csv(os.path.join(DATA, "seeds_dataset.txt"), sep=r"\s+", header=None)
        return df.iloc[:, :7].values.astype(float), df.iloc[:, 7].values
    if name == "Wine":
        d = load_wine()
        return StandardScaler().fit_transform(d.data), d.target
    if name == "Pa":
        df = pd.read_csv(os.path.join(DATA, "parkinsons.data"))
        X = df.drop(columns=["name", "status"]).values.astype(float)
        return StandardScaler().fit_transform(X), df["status"].values
    if name == "BC":
        d = load_breast_cancer()
        return StandardScaler().fit_transform(d.data), d.target
    if name == "Seg":
        seg = pd.read_csv(os.path.join(DATA, "segmentation.test"), skiprows=5, header=None).drop_duplicates()
        y = pd.factorize(seg.iloc[:, 0])[0] + 1
        X = seg.drop([0, 3], axis=1).values.astype(float)   # column 3 (REGION-PIXEL-COUNT) is constant
        return StandardScaler().fit_transform(X), y
    if name == "Yale-B":
        return _mat("yale.mat", "X", "label")
    if name == "Optidigits":
        return _mat("optidigits.mat", "X", "label")
    if name == "Anuran":
        X = sio.loadmat(os.path.join(DATA, "anuran_call.mat"))["X"].astype(float)
        y = sio.loadmat(os.path.join(DATA, "frog_label.mat"))["label"][:, 2]   # species
        return X, y
    if name == "Starbeam":
        return _mat("starbeam.mat", "X", "label")
    if name == "Bainba":
        return _mat("ba.mat", "ba", "ba_label")
    if name == "Blend":
        return _mat("blend.mat", "blend", "blend_label")
    if name == "Agg":
        arr = np.loadtxt(os.path.join(DATA, "aggregation.txt"), delimiter=",")
        return arr[:, :2], arr[:, 2]
    raise ValueError(f"unknown data set {name!r}; choose from {ALL}")
