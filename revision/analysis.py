"""Builds revision tables from the CV results: P/R/F1/error rate per dataset and statistical tests.

Inputs: results/final/*.csv (published runs) + results/final/revision/emnist_digits_rerun.csv
(replaces the two broken EMNIST-Digits rows when present). Outputs to results/final/revision/tables/.
"""
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from common import ROOT

OUT = ROOT / "results/final/revision/tables"
OUT.mkdir(parents=True, exist_ok=True)

DATASETS = ["MNIST", "USPS", "EMNIST-Digits", "Arabic", "Fashion-MNIST"]
CLFS = ["KNN", "MLP", "SVM", "XGBoost", "CatBoost", "LightGBM"]
CLF_LONG = {"KNN": "K-Nearest Neighbors", "MLP": "Multi-Layer Perceptron", "SVM": "Support Vector Machine",
            "XGBoost": "XGBoost", "CatBoost": "CatBoost", "LightGBM": "LightGBM"}
REDS = ["No Reduction", "DFFE (Flood Fill)", "LDA", "PCA", "TruncatedSVD", "PaCMAP", "UMAP"]
RED_SHORT = {"No Reduction": "No DR", "DFFE (Flood Fill)": "DFFE", "TruncatedSVD": "TruncatedSVD"}
DR_ONLY = REDS[1:]


def load():
    a = pd.read_csv(ROOT / "results/final/test_results_summary.csv")
    b = pd.read_csv(ROOT / "results/final/results 2.csv")
    a = a[~((a.classifier_name == "CatBoost") & (a.accuracy > 1))]
    k = ["dataset_name", "reduction_name", "classifier_name"]
    a = a[~a[k].apply(tuple, axis=1).isin(set(map(tuple, b[k].values)))]
    d = pd.concat([a, b])
    rerun = ROOT / "results/final/revision/emnist_digits_rerun.csv"
    if rerun.exists():
        r = pd.read_csv(rerun)
        full = r.groupby(["classifier_name", "reduction_name"]).fold_id.transform("nunique") == 5
        r = r[full]
        d = d[~d[k].apply(tuple, axis=1).isin(set(map(tuple, r[k].values)))]
        d = pd.concat([d, r])
    d = d[d.dataset_name.isin(DATASETS) & d.classifier_name.isin(CLFS) & d.reduction_name.isin(REDS)]
    d["error_rate"] = 1 - d.accuracy
    return d


def latex_metric_table(d, ds, metric, caption):
    g = d[d.dataset_name == ds].groupby(["classifier_name", "reduction_name"])[metric].agg(["mean", "std"]) * 100
    lines = [r"\begin{table*}", r"    \centering",
             rf"    \caption{{{caption}}}", rf"    \label{{tab:{metric}_{ds.lower().replace('-', '_')}}}",
             r"    \resizebox{\textwidth}{!}{%", r"    \begin{tabular}{l|ccccccc}", r"        \hline",
             "        \\textbf{Classifier} & " + " & ".join(rf"\textbf{{{RED_SHORT.get(r, r)}}}" for r in REDS) + r" \\",
             r"        \hline"]
    for c in CLFS:
        cells, best = [], None
        vals = {r: g.loc[(c, r)] if (c, r) in g.index else None for r in REDS}
        dr_means = {r: v["mean"] for r, v in vals.items() if v is not None and r in DR_ONLY}
        if dr_means:
            best = max(dr_means, key=dr_means.get)
        for r in REDS:
            v = vals[r]
            if v is None:
                cells.append("--")
            elif r == best:
                cells.append(rf"\best{{{v['mean']:.2f} \pm {v['std']:.2f}}}")
            else:
                cells.append(rf"${v['mean']:.2f} \pm {v['std']:.2f}$")
        lines.append(f"        {CLF_LONG[c]:22s} & " + " & ".join(cells) + r" \\")
    lines += [r"        \hline", r"    \end{tabular}}", r"\end{table*}", ""]
    return "\n".join(lines)


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def statistics(d, metric="accuracy"):
    """Friedman + Nemenyi (Demsar 2006) over (dataset, classifier) blocks; Wilcoxon DFFE vs others (Holm)."""
    m = d[d.reduction_name.isin(DR_ONLY)].groupby(["dataset_name", "classifier_name", "reduction_name"])[metric] \
        .mean().unstack().dropna()
    m = m[DR_ONLY]
    n, k = m.shape
    ranks = m.rank(axis=1, ascending=False)
    fr = stats.friedmanchisquare(*[m[c] for c in m.columns])
    q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949}[k]
    cd = q05 * np.sqrt(k * (k + 1) / (6 * n))
    rows = []
    others = [c for c in DR_ONLY if c != "DFFE (Flood Fill)"]
    p = []
    for c in others:
        diff = m["DFFE (Flood Fill)"] - m[c]
        w = stats.wilcoxon(m["DFFE (Flood Fill)"], m[c])
        p.append(w.pvalue)
        rows.append({"comparison": f"DFFE vs {c}", "dffe_wins": int((diff > 0).sum()), "losses": int((diff < 0).sum()),
                     "mean_diff_pp": 100 * diff.mean(), "wilcoxon_p": w.pvalue})
    for r, a in zip(rows, holm(np.array(p))):
        r["holm_p"] = a
    summary = {"metric": metric, "n_blocks": n, "k_methods": k, "friedman_chi2": fr.statistic,
               "friedman_p": fr.pvalue, "nemenyi_cd": cd}
    return summary, ranks.mean().sort_values(), pd.DataFrame(rows), m


def corrected_t(d, ds, clf, a, b, metric="accuracy"):
    """Nadeau & Bengio corrected resampled t-test on the 5 paired folds (same splits for all methods)."""
    x = d[(d.dataset_name == ds) & (d.classifier_name == clf)]
    va = x[x.reduction_name == a].sort_values("fold_id")
    vb = x[x.reduction_name == b].sort_values("fold_id")
    if len(va) != 5 or len(vb) != 5:
        return np.nan
    diff = va[metric].values - vb[metric].values
    n_tr, n_te = va.train_set_size.iloc[0], va.test_set_size.iloc[0]
    var = diff.var(ddof=1)
    if var == 0:
        return np.nan
    t = diff.mean() / np.sqrt((1 / 5 + n_te / n_tr) * var)
    return 2 * stats.t.sf(abs(t), df=4)


def main():
    d = load()
    d.to_csv(OUT / "cv_results_merged.csv", index=False)
    tex = []
    for metric, name in [("f1_score", "macro-averaged F1-score"), ("precision", "macro-averaged precision"),
                         ("recall", "macro-averaged recall"), ("error_rate", "error rate")]:
        for ds in DATASETS:
            cap = (f"Classification {name} (\\%) across DR methods for the {ds} dataset. Results reported as "
                   f"mean $\\pm$ std across cross-validation folds.")
            tex.append(latex_metric_table(d, ds, metric, cap))
    (OUT / "metric_tables.tex").write_text("\n".join(tex), encoding="utf-8")

    with open(OUT / "statistics.txt", "w", encoding="utf-8") as f:
        for metric in ["accuracy", "f1_score"]:
            s, ranks, wil, m = statistics(d, metric)
            f.write(f"=== {metric}\n{s}\nAverage ranks (1 = best):\n{ranks.round(3).to_string()}\n")
            f.write(wil.round(5).to_string(index=False) + "\n\n")
            ranks.to_csv(OUT / f"avg_ranks_{metric}.csv")
            wil.to_csv(OUT / f"wilcoxon_{metric}.csv", index=False)
        # per dataset/classifier: DFFE vs each method, corrected resampled t-test
        rows = []
        for ds in DATASETS:
            for c in CLFS:
                for other in [r for r in REDS if r != "DFFE (Flood Fill)"]:
                    rows.append({"dataset": ds, "classifier": c, "other": other,
                                 "p_corrected_t": corrected_t(d, ds, c, "DFFE (Flood Fill)", other)})
        pt = pd.DataFrame(rows)
        pt.to_csv(OUT / "corrected_ttest_dffe_vs_others.csv", index=False)
        f.write("Corrected resampled t-test (DFFE vs other), share of p<0.05: "
                f"{(pt.p_corrected_t < 0.05).mean():.3f}\n")
    print((OUT / "statistics.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
