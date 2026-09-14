# A-DLCC — automatic depth-based local center clustering

Reference implementation of A-DLCC, the parameter-free version of depth-based
local center clustering (DLCC). Given a data matrix the method

1. computes the depth-based similarity matrix and the β-integrated local depth
   (ILD) at every locality level (`adlcc/depth.py`, `adlcc/local_depth.py`);
2. identifies the locally deep points, their self-centrality level B\* and the
   local centers (`adlcc/local_centers.py`, paper Sections 3 and 4.1);
3. forms the cells of the local centers and the group-level local similarity
   (GLS) between cells in contact (`adlcc/similarity.py`, Section 4.1);
4. groups the local centers with the relative-reachability rule — isolation
   check, cells without a boundary, absorption and bonding with the threshold
   Q_th, the community-level contact test, iteration and the reconsideration of
   small groups (`adlcc/grouping.py`, Section 4.2, Algorithm 2);
5. builds the temporary clusters (`adlcc/temp_clusters.py`, Section 4.3);
6. assigns the remaining observations with a depth-based kNN, a random forest
   or the maximum depth classifier (`adlcc/classify.py`).

The only choices are the data and the final classifier; the constants of the
grouping rule (q = 0.975, clip [0.9, 0.99]) are fixed once for all data sets.

## Installation

```
pip install -r requirements.txt
```

Python ≥ 3.10 with NumPy, SciPy, scikit-learn, numba, pandas (data loading)
and matplotlib (figures). PyTorch is optional: it is used for the depth matrix
of data sets with n ≥ 800 when a CUDA GPU is available (the NumPy code gives the
same matrix but is O(n³d) and slow beyond a few thousand points).

## Usage

```python
import numpy as np
from adlcc import adlcc, depth_stage, cluster_performance

X = ...                              # (n, d) array
depth = depth_stage(X)               # depth-based similarity + ILD (the expensive part; cache it)
np.random.seed(2025)                 # only the random forest is random
res = adlcc(X, depth, classifier="rf")   # or "knn", "maxdep"

res["labels"]         # final cluster of every observation, 1..K
res["n_clusters"]     # K
res["local_centers"]  # indices of the local centers (most frequent first)
res["groups"]         # groups of local centers
res["temp_labels"]    # temporary clustering, 0 = assigned only in the final step
```

`adlcc(..., log=print)` prints one line per pairwise grouping decision
(ρ, ω, φ, the verdict and the contact gain), which is the material used for the
discussion of the individual data sets in the paper.

## Reproducing the paper

```
python experiments/run_paper.py                 # every data set of Section 5
python experiments/run_paper.py --datasets Iris Wine Anuran
python experiments/make_figures.py              # Figure 5 panels and the t-SNE figures
```

`run_paper.py` computes the depth stage once per data set (cached under
`experiments/cache/`, ~2.5 GB for all data sets), runs A-DLCC with
`numpy.random.seed(2025)` and, for the data sets whose final step is a random
forest, repeats the final step with seeds 0–99 and reports the mean. The table
printed at the end has the layout of Table 4 of the paper; the values in
`experiments/results/*.json` are the ones reported there (final metrics are the
100-seed means for random-forest data sets and the single deterministic run for
the kNN data sets; the temporary-cluster metrics do not depend on the seed).

Data sets (`experiments/datasets.py`): Iris, Wine and BC from scikit-learn;
Seed and Pa (Parkinsons) are the UCI files; Seg (UCI image segmentation test
file), Yale-B, Optidigits, Anuran Calls and the synthetic Starbeam, Blend,
Bainba and Agg are in `data/`. Wine, Pa, BC and Seg are standardised, as in the
paper.

Run time after the depth stage is seconds for the small data sets, about 20 s
for Seg and Yale-B, 5 min for Optidigits and 10 min for Anuran (the grouping
evaluates reachability on point subsets of a 5620- and 7195-point similarity
matrix). The depth stage itself (RTX 3060, float64) takes 20–30 s for Seg and
Blend, 4 min for Yale-B (d = 600), 5 min for Optidigits and 7 min for Anuran;
with NumPy the small data sets take under a minute.

### Determinism

Everything except the random forest is deterministic. The NumPy and PyTorch
depth codes agree to about 1e-15; this can change the neighbour order only when
the data contain exact ties (Iris has duplicate observations). The paper's
results were computed with NumPy for n < 800 and the GPU otherwise, which is the
default rule of `depth_similarity`; pass `--device numpy` or `--device cuda` to
force one implementation.

## Implementation notes (details fixed in the code, not spelled out in the paper)

* Locality levels: neighbourhood sizes run from min(2d, 10) to n in steps of one.
* Self-centrality (`local_centers.self_centrality`): as in the original R code,
  the first 20 levels are not counted when forming B(x) (an ILD averaged over
  fewer than 20 levels is not considered stable), and B(x) must contain at least
  two levels for x to be a candidate.
* Stability condition of a local center: proportion ≥ 0.5 (the paper writes > 0.5).
* Cells of a single observation are dissolved before the GLS is computed.
* The depth-based kNN classifier uses the (unsymmetrised) depth-based
  similarity matrix, as in DLCC; the random forest is scikit-learn's
  `RandomForestClassifier(n_estimators=100, min_samples_leaf=3)`, equivalent to
  R's `randomForest(ntree=100, nodesize=3)`.

## Layout

```
adlcc/                 the method
  depth.py             spatial depth, depth-based similarity matrix (NumPy / PyTorch)
  local_depth.py       β-local depth and β-integrated local depth
  local_centers.py     locally deep points, self-centrality level, local centers
  similarity.py        cells, GLS, reachable similarity, isolation check
  grouping.py          adaptive grouping of the local centers (Algorithm 2)
  temp_clusters.py     temporary clusters
  classify.py          final assignment (rf / knn / maxdep)
  metrics.py           ARI, NMI, purity
  pipeline.py          depth_stage() and adlcc()
experiments/
  datasets.py          the data sets of Section 5 and their classifiers
  run_paper.py         Table 4 / Anuran table numbers (with 100-seed RF means)
  make_figures.py      Figure 5 panels and t-SNE figures
  results/             json + npz written by run_paper.py
data/                  data files used by experiments/datasets.py
```
