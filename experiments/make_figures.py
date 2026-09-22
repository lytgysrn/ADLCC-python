"""Figures of the paper from the saved results of ``run_paper.py``.

    python experiments/make_figures.py

* Figure 5, A-DLCC panels: scatter plots of Starbeam, Blend, Bainba and Agg
  coloured by the A-DLCC clusters.
* Figures 6 and 7: the same three-panel t-SNE rows for Iris, Wine, Seed
  and for Pa, BC, stacked into ``NEWPAPER/combine_1.png`` and ``combine_2.png``.
* Figures 8 and 9: t-SNE embeddings of Seg, Yale-B, Optidigits and Anuran with
  three panels each: ground truth, temporary clusters with the local centers
  (triangles), and the A-DLCC result.
Written to ``experiments/figures/``.
"""
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets import load_dataset  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGURES = os.path.join(HERE, "figures")
PAL = ["#F8766D", "#A3A500", "#00BF7D", "#00B0F6", "#E76BF3", "#D89000", "#39B600", "#00BFC4", "#9590FF", "#FF62BC",
       "#B79F00", "#619CFF", "#7CAE00", "#C77CFF", "#00A9FF", "#FF61C3", "#8494FF", "#E68613", "#0CB702", "#ED68ED"]


def _style(ax):
    ax.set_facecolor("white")
    ax.grid(True, color="#EBEBEB", linewidth=0.8)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_visible(False)


def _scatter(ax, emb, lab, size, lcs=None):
    for i, l in enumerate(np.unique(lab)):
        m = lab == l
        ax.scatter(emb[m, 0], emb[m, 1], s=size, color="black" if l == 0 else PAL[i % len(PAL)], linewidths=0)
    if lcs is not None:
        ax.scatter(emb[lcs, 0], emb[lcs, 1], s=40, marker="^", color="black", zorder=5)


def synthetic_panel(name):
    X, _ = load_dataset(name)
    r = np.load(os.path.join(RESULTS, f"{name}.npz"))
    fig, ax = plt.subplots(figsize=(7.5, 3.5) if name == "Agg" else (5.2, 4.2))
    _style(ax)
    _scatter(ax, X, r["labels"], 9)
    ax.set_xlabel("V1", fontsize=9)
    ax.set_ylabel("V2", fontsize=9)
    ax.tick_params(labelsize=7, length=0)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, f"{name.lower()}_adlcc.png"), dpi=150)
    plt.close(fig)


def tsne_row(axes, name, emb, y, r, size):
    """One data set, three panels, same layout as ``tsne_panels``."""
    for ax in axes:
        _style(ax)
        ax.set_xticks([])
        ax.set_yticks([])
    _scatter(axes[0], emb, np.asarray(y).ravel() + 1, size)
    axes[0].set_title(f"Ground truth ($K={len(np.unique(y))}$)", fontsize=10)
    axes[0].set_ylabel(name, fontsize=11)
    _scatter(axes[1], emb, r["temp_labels"], size, lcs=r["local_centers"])
    axes[1].set_title("Temporary clusters and local centers", fontsize=10)
    _scatter(axes[2], emb, r["labels"], size)
    axes[2].set_title(f"A-DLCC ($\\hat K={len(np.unique(r['labels']))}$)", fontsize=10)


def stacked_tsne(names, path):
    """Several data sets, one three-panel row each, written as a single figure."""
    rows = []
    for name in names:
        X, y = load_dataset(name)
        r = np.load(os.path.join(RESULTS, f"{name}.npz"))
        emb = TSNE(n_components=2, random_state=2025, init="pca", perplexity=30).fit_transform(np.asarray(X, float))
        rows.append((name, emb, np.asarray(y).ravel(), r))
    fig, axes = plt.subplots(len(names), 3, figsize=(13, 4.0 * len(names)))
    if len(names) == 1:
        axes = np.array([axes])
    for i, (name, emb, y, r) in enumerate(rows):
        tsne_row(axes[i], name, emb, y, r, 8 if len(y) < 800 else 4)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def tsne_panels(name):
    X, y = load_dataset(name)
    y = np.asarray(y).ravel()
    r = np.load(os.path.join(RESULTS, f"{name}.npz"))
    emb = TSNE(n_components=2, random_state=2025, init="pca", perplexity=30).fit_transform(np.asarray(X, float))
    size = 3 if name == "Anuran" else 4
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax in axes:
        _style(ax)
        ax.set_xticks([])
        ax.set_yticks([])
    _scatter(axes[0], emb, y + 1, size)
    axes[0].set_title(f"Ground truth ($K={len(np.unique(y))}$)", fontsize=10)
    _scatter(axes[1], emb, r["temp_labels"], size, lcs=r["local_centers"])
    axes[1].set_title("Temporary clusters and local centers", fontsize=10)
    _scatter(axes[2], emb, r["labels"], size)
    axes[2].set_title(f"A-DLCC ($\\hat K={len(np.unique(r['labels']))}$)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, f"{name.lower().replace('-', '')}_tsne.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FIGURES, exist_ok=True)
    for name in ["Starbeam", "Blend", "Bainba", "Agg"]:
        if os.path.exists(os.path.join(RESULTS, f"{name}.npz")):
            synthetic_panel(name)
            print("wrote", f"{name.lower()}_adlcc.png", flush=True)
    paper = os.path.normpath(os.path.join(HERE, "..", "..", "NEWPAPER"))
    stacked_tsne(["Iris", "Wine", "Seed"], os.path.join(paper, "combine_1.png"))
    print("wrote combine_1.png", flush=True)
    stacked_tsne(["Pa", "BC"], os.path.join(paper, "combine_2.png"))
    print("wrote combine_2.png", flush=True)
    for name in ["Seg", "Yale-B", "Optidigits", "Anuran"]:
        if os.path.exists(os.path.join(RESULTS, f"{name}.npz")):
            tsne_panels(name)
            print("wrote", f"{name.lower().replace('-', '')}_tsne.png", flush=True)
