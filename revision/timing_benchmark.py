"""Computational cost benchmark (reviewer requests: time reduction due to DR, training/testing time,
complexity, memory, scalability). Run alone on an idle machine.

A) DR cost on MNIST fold 1 (48k train / 12k val): fit on train, transform the unseen val samples,
   per-sample latency, size of the fitted reducer (what must be stored to embed new samples).
B) Classifier training / prediction time on each representation (protocol features, fold 1).
C) Scalability of DR fit+transform with the number of samples (EMNIST-Digits subsets).
D) DFFE extraction time per image vs image resolution (linear in m*n).
"""
import pickle
import platform
import time

import numpy as np
import pandas as pd

from common import (N_COMPONENTS, PIXEL_THRESHOLD, REDUCTIONS, ROOT, build_features, evaluate_fold,
                    folds, load_dataset)
from BFS.bfs import calculate_flooded_vector

OUT = ROOT / "results/final/revision"
OUT.mkdir(parents=True, exist_ok=True)
CLASSIFIERS = ["KNN", "SVM", "MLP", "XGBoost", "CatBoost", "LightGBM"]


def dffe(binar):
    return np.asarray([calculate_flooded_vector(b, num_segments=7, floodSides="1111") for b in binar])


def make_reducer(red, nc):
    if red == "PCA":
        from sklearn.decomposition import PCA
        return PCA(n_components=N_COMPONENTS)
    if red == "TruncatedSVD":
        from sklearn.decomposition import TruncatedSVD
        return TruncatedSVD(n_components=N_COMPONENTS, random_state=42)
    if red == "LDA":
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        return LinearDiscriminantAnalysis(n_components=nc - 1)
    if red == "UMAP":
        import umap
        return umap.UMAP(n_components=N_COMPONENTS, random_state=42)
    if red == "PaCMAP":
        import pacmap
        return pacmap.PaCMAP(n_components=N_COMPONENTS, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0,
                             random_state=42, save_tree=True)
    raise ValueError(red)


def fit_transform_cost(red, X_tr, y_tr, X_new, nc, binar_tr=None, binar_new=None):
    """Returns (fit_s, transform_s, model_bytes)."""
    if red == "No Reduction":
        t0 = time.perf_counter(); binar_new.reshape(len(binar_new), -1).astype(np.float64)
        return 0.0, time.perf_counter() - t0, 0
    if red == "DFFE (Flood Fill)":
        t0 = time.perf_counter(); dffe(binar_new)
        return 0.0, time.perf_counter() - t0, 0
    reducer = make_reducer(red, nc)
    t0 = time.perf_counter()
    if red == "LDA":
        reducer.fit(X_tr, y_tr)
    elif red == "PaCMAP":
        reducer.fit(X_tr, init="pca")
    else:
        reducer.fit(X_tr)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    if red == "PaCMAP":
        reducer.transform(X_new, basis=X_tr)
    else:
        reducer.transform(X_new)
    tr_s = time.perf_counter() - t0
    try:
        size = len(pickle.dumps(reducer))
    except Exception:
        size = -1
    if red == "PaCMAP":  # transform needs the training set itself as the basis
        size += X_tr.nbytes
    return fit_s, tr_s, size


def main():
    info = {"cpu": platform.processor(), "python": platform.python_version()}
    print(info, flush=True)
    dffe(np.zeros((2, 28, 28), dtype=np.int64))  # numba JIT warm-up, excluded from timings

    # ---------------- A) DR cost, MNIST fold 1
    pix, y, size, nc = load_dataset("MNIST")
    tr, va = next(iter(folds(y)))
    binar = np.where(pix > PIXEL_THRESHOLD, 1, 0).reshape(-1, size, size)
    rows = []
    for red in REDUCTIONS:
        reps = 1 if red in ("UMAP", "PaCMAP") else 3
        for rep in range(reps):
            try:
                fit_s, tr_s, nbytes = fit_transform_cost(red, pix[tr], y[tr], pix[va], nc, binar[tr], binar[va])
            except Exception as e:
                print(f"A {red} FAILED: {e}", flush=True)
                continue
            rows.append({"reduction_name": red, "repeat": rep, "fit_time_s": fit_s, "transform_time_s": tr_s,
                         "n_new_samples": len(va), "per_sample_ms": 1000 * tr_s / len(va),
                         "stored_model_bytes": nbytes})
            print(f"A {red:18s} fit={fit_s:8.2f}s transform={tr_s:7.2f}s model={nbytes/1e6:8.2f}MB", flush=True)
            pd.DataFrame(rows).to_csv(OUT / "timing_dr_mnist.csv", index=False)

    # ---------------- B) classifier cost per representation, MNIST fold 1
    rows = []
    for red in REDUCTIONS:
        X, _ = build_features(pix, y, red, size, nc)
        for clf in CLASSIFIERS:
            try:
                r = evaluate_fold(clf, X[tr], y[tr], X[va], y[va], nc)
            except Exception as e:
                print(f"B {red} {clf} FAILED: {e}", flush=True)
                continue
            rows.append({"reduction_name": red, "classifier_name": clf, "n_features": X.shape[1],
                         "feature_matrix_MB_float64": X.nbytes / 1e6, **r})
            print(f"B {red:18s} {clf:9s} train={r['training_time']:8.2f}s test={r['testing_time']:7.2f}s "
                  f"acc={r['accuracy']:.4f}", flush=True)
            pd.DataFrame(rows).to_csv(OUT / "timing_classifiers_mnist.csv", index=False)

    # ---------------- D) DFFE time vs resolution
    rows = []
    sample = binar[:2000]
    for k in (1, 2, 4, 8):
        imgs = np.kron(sample, np.ones((1, k, k), dtype=sample.dtype))
        dffe(imgs[:2])
        t0 = time.perf_counter(); dffe(imgs)
        dt = time.perf_counter() - t0
        rows.append({"side_px": 28 * k, "pixels": (28 * k) ** 2, "n_images": len(imgs),
                     "per_image_ms": 1000 * dt / len(imgs)})
        print(f"D side={28*k:4d} per_image={1000*dt/len(imgs):.3f}ms", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "dffe_resolution.csv", index=False)

    # ---------------- C) scalability with number of samples, EMNIST-Digits
    pix, y, size, nc = load_dataset("EMNIST-Digits")
    binar = np.where(pix > PIXEL_THRESHOLD, 1, 0).reshape(-1, size, size)
    rng = np.random.default_rng(42)
    order = rng.permutation(len(y))
    rows = []
    for n in (10_000, 30_000, 60_000, 120_000, 240_000):
        idx = order[:n]
        for red in ["DFFE (Flood Fill)", "PCA", "TruncatedSVD", "LDA", "UMAP", "PaCMAP"]:
            t0 = time.perf_counter()
            try:
                if red == "DFFE (Flood Fill)":
                    dffe(binar[idx])
                else:
                    m = make_reducer(red, nc)
                    if red == "LDA":
                        m.fit_transform(pix[idx], y[idx])
                    elif red == "PaCMAP":
                        m.fit_transform(pix[idx], init="pca")
                    else:
                        m.fit_transform(pix[idx])
            except Exception as e:
                print(f"C n={n} {red} FAILED: {e}", flush=True)
                continue
            dt = time.perf_counter() - t0
            rows.append({"n_samples": n, "reduction_name": red, "fit_transform_time_s": dt})
            print(f"C n={n:7d} {red:18s} {dt:9.2f}s", flush=True)
            pd.DataFrame(rows).to_csv(OUT / "scalability_emnist.csv", index=False)


if __name__ == "__main__":
    main()
