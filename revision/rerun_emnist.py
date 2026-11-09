"""Re-run of the two broken EMNIST-Digits rows (CatBoost: no valid source; LightGBM: diverged folds).

Same protocol as the published tables; additionally records per-fold training/testing time.
LightGBM uses the hessian floor from common.LGBM_STABLE (default setting diverges on this dataset;
the diverged default-setting folds are kept in emnist_lightgbm_default_diverged.csv).
`python rerun_emnist.py --mnist-lgbm-check` runs the stabilised LightGBM on MNIST to show that the
setting does not change results on datasets where the default converges.
Resumable: rows already present in the output CSV are skipped.
"""
import sys

import pandas as pd

from common import REDUCTIONS, ROOT, build_features, evaluate_fold, folds, load_dataset


def run(ds, classifiers, out, lgbm_stable=True):
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        prev = pd.read_csv(out).drop_duplicates(["classifier_name", "reduction_name", "fold_id"], keep="last")
        prev.to_csv(out, index=False)
        done = set(zip(prev.classifier_name, prev.reduction_name, prev.fold_id))
    pix, y, size, nc = load_dataset(ds)
    split = list(folds(y))
    for red in REDUCTIONS:
        if all((c, red, f + 1) in done for c in classifiers for f in range(len(split))):
            continue
        X, red_time = build_features(pix, y, red, size, nc)
        print(f"[{red}] features {X.shape} in {red_time:.1f}s", flush=True)
        for clf in classifiers:
            for f, (tr, va) in enumerate(split, start=1):
                if (clf, red, f) in done:
                    continue
                stable = lgbm_stable and clf == "LightGBM"
                r = evaluate_fold(clf, X[tr], y[tr], X[va], y[va], nc, lgbm_stable=stable)
                row = {"dataset_name": ds, "classifier_name": clf, "reduction_name": red, "fold_id": f,
                       "n_components": X.shape[1], "reduction_time_full_train": red_time,
                       "lgbm_min_sum_hessian_in_leaf": 1.0 if stable else None, **r}
                pd.DataFrame([row]).to_csv(out, mode="a", header=not out.exists(), index=False)
                print(f"  {clf:9s} fold {f}: acc={r['accuracy']:.4f} train={r['training_time']:.1f}s "
                      f"test={r['testing_time']:.2f}s", flush=True)


if __name__ == "__main__":
    if "--mnist-lgbm-check" in sys.argv:
        run("MNIST", ["LightGBM"], ROOT / "results/final/revision/mnist_lightgbm_stable_check.csv")
    else:
        run("EMNIST-Digits", ["LightGBM", "CatBoost"], ROOT / "results/final/revision/emnist_digits_rerun.csv")
