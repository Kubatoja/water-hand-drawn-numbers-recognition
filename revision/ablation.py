"""Ablation study of the DFFE components (reviewer request).

Feature layout produced by calculate_flooded_vector (N segments, flood "1111"):
  [0:N) left | [N:2N) right | [2N:3N) top | [3N:4N) bottom   directional floods (minus inner area)
  [4N:5N) inner area | [5N:6N) pixel density | [6N] perimeter
Variants either drop feature groups from the full vector or recompute the vector with a
changed component (no inner-area correction, number of segments N, binarisation threshold).
Same 5-fold protocol and classifier settings as the published tables.
"""
import sys

import numpy as np
import pandas as pd

from common import NUM_SEGMENTS, PIXEL_THRESHOLD, ROOT, evaluate_fold, folds, load_dataset
from BFS.bfs import (calculate_flooded_vector, calculate_segments_numba,
                     flood_from_all_sides_numba)

OUT = ROOT / "results/final/revision/ablation.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)
DATASETS = sys.argv[1:] or ["MNIST", "Fashion-MNIST", "USPS"]
CLASSIFIERS = ["KNN", "LightGBM"]


def dffe(binar, n_seg=NUM_SEGMENTS):
    return np.asarray([calculate_flooded_vector(b, num_segments=n_seg, floodSides="1111") for b in binar])


def dffe_no_correction(binar, n_seg=NUM_SEGMENTS):
    """Directional floods WITHOUT subtracting the inner (enclosed) area; other groups unchanged."""
    rows = []
    for b in binar:
        a = b.astype(np.float64)
        l, r, t, bo, _ = flood_from_all_sides_numba(a)
        dirs = [calculate_segments_numba((m == 0).astype(np.float64), n_seg) for m in (l, r, t, bo)]
        rest = calculate_flooded_vector(b, num_segments=n_seg, floodSides="0000")  # inner, density, perimeter
        rows.append(np.concatenate(dirs + [np.asarray(rest)]))
    return np.asarray(rows)


def groups(n):
    g = {"L": range(0, n), "R": range(n, 2 * n), "T": range(2 * n, 3 * n), "B": range(3 * n, 4 * n),
         "inner": range(4 * n, 5 * n), "density": range(5 * n, 6 * n), "perimeter": [6 * n]}
    return {k: list(v) for k, v in g.items()}


def subset_variants(n):
    g = groups(n)
    allg = ["L", "R", "T", "B", "inner", "density", "perimeter"]
    sel = lambda names: sorted(i for k in names for i in g[k])  # noqa: E731
    return {
        "Full DFFE": sel(allg),
        "w/o perimeter": sel([k for k in allg if k != "perimeter"]),
        "w/o pixel density": sel([k for k in allg if k != "density"]),
        "w/o inner area": sel([k for k in allg if k != "inner"]),
        "w/o directional floods": sel(["inner", "density", "perimeter"]),
        "Directional floods only": sel(["L", "R", "T", "B"]),
        "Pixel density only": sel(["density"]),
        "Horizontal floods only (L+R)": sel(["L", "R", "inner", "density", "perimeter"]),
        "Vertical floods only (T+B)": sel(["T", "B", "inner", "density", "perimeter"]),
        "Single flood direction (L)": sel(["L", "inner", "density", "perimeter"]),
    }


done = set()
if OUT.exists():  # resume: keep only complete (all folds) groups, drop partial ones
    prev = pd.read_csv(OUT)
    n = prev.groupby(["dataset_name", "variant", "classifier_name"]).fold_id.transform("nunique")
    prev = prev[n == 5]
    prev.to_csv(OUT, index=False)
    done = set(zip(prev.dataset_name, prev.variant, prev.classifier_name))


def run(ds, variant, X, y, nc):
    for clf in CLASSIFIERS:
        if (ds, variant, clf) in done:
            continue
        for f, (tr, va) in enumerate(folds(y), start=1):
            r = evaluate_fold(clf, X[tr], y[tr], X[va], y[va], nc)
            pd.DataFrame([{"dataset_name": ds, "variant": variant, "classifier_name": clf, "fold_id": f,
                           "n_features": X.shape[1], **r}]).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
        print(f"{ds:14s} {variant:32s} {clf:8s} dim={X.shape[1]:3d} done", flush=True)


for ds in DATASETS:
    pix, y, size, nc = load_dataset(ds)
    binar = np.where(pix > PIXEL_THRESHOLD, 1, 0).reshape(-1, size, size)
    full = dffe(binar)
    for name, cols in subset_variants(NUM_SEGMENTS).items():
        run(ds, name, full[:, cols], y, nc)
    run(ds, "No inner-area correction of floods", dffe_no_correction(binar), y, nc)
    for n in (3, 5, 10):
        run(ds, f"N = {n} segments", dffe(binar, n), y, nc)
    for tau in (0.1, 0.5):
        run(ds, f"Threshold tau = {tau}", dffe(np.where(pix > tau, 1, 0).reshape(-1, size, size)), y, nc)
