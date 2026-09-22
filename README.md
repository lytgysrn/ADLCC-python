# A-DLCC

This repository provides the Python code for **Automatic Depth-based Local Center Clustering (A-DLCC)** via β-integrated local depth and adaptive grouping.

> **Note:**
> Cursor assisted with the Python code.

Given a data matrix, the method computes the depth-based similarity and the β-integrated local depth, finds the local centers and groups them by the relative-reachability rule, builds temporary clusters, and assigns the remaining observations with a depth-based kNN, a random forest, or the maximum depth classifier. The only choices are the data and the final classifier.

## Installation

```
pip install -r requirements.txt
```

Python ≥ 3.10 with NumPy, SciPy, scikit-learn, numba, pandas and matplotlib. PyTorch is optional: it is used for the depth matrix when n ≥ 800 and a CUDA GPU is available. The NumPy code gives the same matrix.

## Usage

```python
import numpy as np
from adlcc import adlcc, depth_stage

X = ...                              # (n, d) array
depth = depth_stage(X)               # depth-based similarity + ILD; cache this
np.random.seed(2025)                 # only the random forest is random
res = adlcc(X, depth, classifier="rf")   # or "knn", "maxdep"

res["labels"]         # final cluster of every observation, 1..K
res["n_clusters"]
res["local_centers"]  # indices of the local centers
res["groups"]         # groups of local centers
res["temp_labels"]    # temporary clustering, 0 = assigned only in the final step
```

## Reproducing the paper

```
python experiments/run_paper.py
python experiments/run_paper.py --datasets Iris Wine Anuran
python experiments/make_figures.py
```

`run_paper.py` computes the depth stage once per data set (cached under `experiments/cache/`) and, for random-forest data sets, repeats the final step with seeds 0–99 and reports the mean. The values in `experiments/results/` are those reported in the paper. Iris, Wine and BC come from scikit-learn; the other files are in `data/`. Wine, Pa, BC and Seg are standardised.

## Structure

* `adlcc/`: depth, local centers, grouping, temporary clusters, classification
* `experiments/`: `run_paper.py`, `make_figures.py`, and the saved results
* `data/`: data files used by the experiments

Other datasets used with the original DLCC code: [https://github.com/lytgysrn/dlcc](https://github.com/lytgysrn/dlcc)

---

**For questions or suggestions, please contact the authors.**
