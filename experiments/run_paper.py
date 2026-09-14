"""Reproduce the A-DLCC numbers of the paper (Table 4, the Anuran table, Figure 5).

    python experiments/run_paper.py                       # all data sets
    python experiments/run_paper.py --datasets Iris Wine  # a selection
    python experiments/run_paper.py --device numpy        # force the NumPy depth code

For every data set the script
  1. computes (or loads from ``experiments/cache/``) the depth-based similarity
     matrix and the integrated local depth,
  2. runs A-DLCC once with ``numpy.random.seed(2025)`` and reports K-hat, the
     coverage of the temporary clusters and ARI / Purity / NMI of the temporary
     and of the final clustering,
  3. for the data sets whose final step is a random forest, repeats the final
     step with seeds 0..N-1 (default N = 100) and reports the mean and standard
     deviation over seeds; these means are the values printed in the paper.
Results are written to ``experiments/results/<name>.json`` (numbers) and
``<name>.npz`` (label vectors, used by ``make_figures.py``), and a summary
table is printed at the end.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from adlcc.pipeline import depth_stage, adlcc               # noqa: E402
from adlcc.classify import assign_remaining                  # noqa: E402
from adlcc.metrics import cluster_performance                # noqa: E402
from datasets import load_dataset, CLASSIFIER, ALL           # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
RESULTS = os.path.join(HERE, "results")
SEED = 2025


def cached_depth(name, X, device=None):
    """depth_stage with an on-disk cache (the depth matrix is the expensive part)."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{name}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return {k: z[k] for k in z.files}
    t0 = time.time()
    d = depth_stage(X, device=device)
    d = {k: d[k] for k in ("dm", "ILD_mat", "dm0_order", "nbr_list")}
    np.savez(path, **d)
    print(f"  depth + ILD computed in {time.time() - t0:.1f}s", flush=True)
    return d


def evaluate(y, labels):
    m = labels != 0
    perf = cluster_performance(y[m], labels[m]) if m.any() else {"ARI": 0.0, "NMI": 0.0, "Purity": 0.0}
    perf["K"] = int(len(np.unique(labels[m]))) if m.any() else 0
    perf["coverage"] = float(m.mean() * 100)
    return perf


def run_one(name, n_seeds, device=None):
    X, y = load_dataset(name)
    y = np.asarray(y).ravel()
    clf = CLASSIFIER[name]
    print(f"== {name}: n={X.shape[0]} d={X.shape[1]} K={len(np.unique(y))} classifier={clf}", flush=True)
    depth = cached_depth(name, X, device=device)

    t0 = time.time()
    np.random.seed(SEED)
    res = adlcc(X, depth, classifier=clf)
    took = time.time() - t0
    temp, final = evaluate(y, res["temp_labels"]), evaluate(y, res["labels"])
    rec = {
        "name": name, "n": int(X.shape[0]), "d": int(X.shape[1]), "true_K": int(len(np.unique(y))),
        "classifier": clf, "n_local_centers": int(len(res["local_centers"])),
        "groups": [[int(c) for c in g] for g in res["groups"]],
        "K": final["K"], "coverage": round(temp["coverage"], 1),
        "temp": {k: round(temp[k], 6) for k in ("ARI", "Purity", "NMI")},
        "final_seed2025": {k: round(final[k], 6) for k in ("ARI", "Purity", "NMI")},
        "seconds_after_depth": round(took, 2),
    }
    line = (f"  K={final['K']} (temp cov {temp['coverage']:.1f}%)  "
            f"final ARI={final['ARI']:.3f} Pur={final['Purity']:.3f} NMI={final['NMI']:.3f}  "
            f"temp ARI={temp['ARI']:.3f} Pur={temp['Purity']:.3f} NMI={temp['NMI']:.3f}  ({took:.1f}s)")
    if clf == "rf" and n_seeds > 0:
        vals = {"ARI": [], "Purity": [], "NMI": []}
        for s in range(n_seeds):
            np.random.seed(s)
            lab = assign_remaining(X, res["temp_clus"], method="rf", dm=depth["dm"])
            p = cluster_performance(y, lab)
            for k in vals:
                vals[k].append(p[k])
        rec["final_rf_seeds"] = {
            "n_seeds": n_seeds,
            **{k: {"mean": round(float(np.mean(v)), 6), "sd": round(float(np.std(v)), 6),
                   "min": round(float(np.min(v)), 6), "max": round(float(np.max(v)), 6)} for k, v in vals.items()},
        }
        r = rec["final_rf_seeds"]
        line += (f"\n  RF {n_seeds} seeds: ARI {r['ARI']['mean']:.3f}+-{r['ARI']['sd']:.3f}  "
                 f"Pur {r['Purity']['mean']:.3f}  NMI {r['NMI']['mean']:.3f}")
    print(line, flush=True)

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1)
    np.savez(os.path.join(RESULTS, f"{name}.npz"), labels=res["labels"], temp_labels=res["temp_labels"],
             local_centers=np.asarray(res["local_centers"], int), y=y)
    return rec


def summary(names):
    lines = [f"{'data set':11s} {'clf':4s} {'K':>3s} {'cov%':>6s} | {'ARI':>6s} {'Purity':>6s} {'NMI':>6s} | "
             f"{'ARI_t':>6s} {'Pur_t':>6s} {'NMI_t':>6s} | reported"]
    for name in names:
        path = os.path.join(RESULTS, f"{name}.json")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
        fin = r["final_seed2025"]
        rep = "seed 2025"
        if "final_rf_seeds" in r:
            fin = {k: r["final_rf_seeds"][k]["mean"] for k in ("ARI", "Purity", "NMI")}
            rep = f"mean of {r['final_rf_seeds']['n_seeds']} RF seeds"
        t = r["temp"]
        lines.append(f"{name:11s} {r['classifier']:4s} {r['K']:3d} {r['coverage']:6.1f} | "
                     f"{fin['ARI']:6.3f} {fin['Purity']:6.3f} {fin['NMI']:6.3f} | "
                     f"{t['ARI']:6.3f} {t['Purity']:6.3f} {t['NMI']:6.3f} | {rep}")
    text = "\n".join(lines)
    print("\n" + "=" * 96 + "\n" + text)
    if set(names) == set(ALL):
        with open(os.path.join(RESULTS, "summary.txt"), "w", encoding="utf-8") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=ALL, help=f"any of {ALL}")
    ap.add_argument("--seeds", type=int, default=100, help="number of random-forest seeds (0 = single run only)")
    ap.add_argument("--device", default=None,
                    help="'numpy', 'cuda' or 'cpu' (PyTorch); default: numpy for n < 800, else cuda if available")
    ap.add_argument("--summary-only", action="store_true", help="only print the table from experiments/results/")
    args = ap.parse_args()
    if not args.summary_only:
        for name in args.datasets:
            run_one(name, args.seeds, device=args.device)
    summary(args.datasets)
