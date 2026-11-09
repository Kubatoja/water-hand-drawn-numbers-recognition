"""Sanity check: reproduce published fold-level KNN results before running new experiments."""
import sys

import pandas as pd

from common import ROOT, build_features, evaluate_fold, folds, load_dataset

ref = pd.concat([pd.read_csv(ROOT / "results/final/test_results_summary.csv"),
                 pd.read_csv(ROOT / "results/final/results 2.csv")])

for ds in sys.argv[1:]:
    pix, y, size, nc = load_dataset(ds)
    for red in (["DFFE (Flood Fill)"] if ds == "EMNIST-Digits" else ["DFFE (Flood Fill)", "PCA"]):
        X, _ = build_features(pix, y, red, size, nc)
        got = [evaluate_fold("KNN", X[tr], y[tr], X[va], y[va], nc)["accuracy"] for tr, va in folds(y)]
        exp = ref[(ref.dataset_name == ds) & (ref.classifier_name == "KNN") & (ref.reduction_name == red)] \
            .sort_values("fold_id").accuracy.tolist()
        print(f"{ds:15s} {red:18s} new={['%.4f' % a for a in got]}\n{'':34s} old={['%.4f' % a for a in exp]}", flush=True)
