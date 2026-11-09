"""LaTeX tables + plain summary from the revision runs (ablation, timing, scalability, EMNIST rerun)."""
import pandas as pd

from common import ROOT

R = ROOT / "results/final/revision"
OUT = R / "tables"
OUT.mkdir(parents=True, exist_ok=True)
SHORT = {"No Reduction": "No DR", "DFFE (Flood Fill)": "DFFE"}
ORDER = ["No Reduction", "DFFE (Flood Fill)", "LDA", "PCA", "TruncatedSVD", "PaCMAP", "UMAP"]
CLFS = ["KNN", "SVM", "MLP", "XGBoost", "CatBoost", "LightGBM"]


def tab(caption, label, header, rows, colspec):
    return "\n".join([r"\begin{table*}", r"    \centering", rf"    \caption{{{caption}}}", rf"    \label{{{label}}}",
                      r"    \resizebox{\textwidth}{!}{%", rf"    \begin{{tabular}}{{{colspec}}}", r"        \hline",
                      "        " + " & ".join(header) + r" \\", r"        \hline"]
                     + ["        " + " & ".join(r) + r" \\" for r in rows]
                     + [r"        \hline", r"    \end{tabular}}", r"\end{table*}", ""])


parts, summary = [], []

# ---- ablation
a = pd.read_csv(R / "ablation.csv")
g = a.groupby(["variant", "dataset_name", "classifier_name"], sort=False).agg(
    m=("accuracy", "mean"), s=("accuracy", "std"), dim=("n_features", "first"))
variants = list(dict.fromkeys(a.variant))
cols = [(d, c) for d in ["MNIST", "Fashion-MNIST", "USPS"] for c in ["KNN", "LightGBM"]]
rows = []
for v in variants:
    dim = int(g.loc[v].dim.iloc[0]) if v in g.index.get_level_values(0) else 0
    dims = sorted(set(int(x) for x in g.loc[v].dim))
    rows.append([v, "/".join(map(str, dims))] + [f"${100*g.loc[(v,)+c].m:.2f} \\pm {100*g.loc[(v,)+c].s:.2f}$" for c in cols])
parts.append(tab("Ablation study of the DFFE components: classification accuracy (\\%, mean $\\pm$ std over 5 CV folds). "
                 "Dim. denotes the resulting feature-vector length.", "tab:ablation",
                 ["\\textbf{Variant}", "\\textbf{Dim.}"] + [f"\\textbf{{{d}}} {c}" for d, c in cols], rows, "l|c|cccccc"))

# ---- DR cost (MNIST fold 1)
t = pd.read_csv(R / "timing_dr_mnist.csv").groupby("reduction_name", sort=False).agg(
    fit=("fit_time_s", "median"), tr=("transform_time_s", "median"), ps=("per_sample_ms", "median"),
    mb=("stored_model_bytes", "first"))
rows = [[SHORT.get(r, r), f"{t.loc[r].fit:.2f}", f"{t.loc[r].tr:.2f}", f"{t.loc[r].ps:.3f}", f"{t.loc[r].mb/1e6:.2f}"]
        for r in ORDER if r in t.index]
parts.append(tab("Cost of the DR step on MNIST (fit on 48{,}000 training images, embedding of 12{,}000 unseen images; "
                 "median of 3 runs, single run for UMAP and PaCMAP). Model size is the memory that must be kept to embed new samples.",
                 "tab:dr_cost", ["\\textbf{Method}", "\\textbf{Fit [s]}", "\\textbf{Transform [s]}",
                                 "\\textbf{Per sample [ms]}", "\\textbf{Stored model [MB]}"], rows, "l|cccc"))

# ---- classifier training / testing time
c = pd.read_csv(R / "timing_classifiers_mnist.csv").set_index(["reduction_name", "classifier_name"])
rows = []
for clf in CLFS:
    rows.append([clf] + [f"{c.loc[(r, clf)].training_time:.1f} / {c.loc[(r, clf)].testing_time:.2f}" for r in ORDER])
parts.append(tab("Classifier training / testing time [s] on MNIST (48{,}000 training, 12{,}000 test samples, CV fold 1) "
                 "for each representation.", "tab:clf_time",
                 ["\\textbf{Classifier}"] + [f"\\textbf{{{SHORT.get(r, r)}}}" for r in ORDER], rows, "l|ccccccc"))

# ---- scalability
s = pd.read_csv(R / "scalability_emnist.csv").pivot(index="n_samples", columns="reduction_name", values="fit_transform_time_s")
meths = [m for m in ORDER if m in s.columns]
rows = [[f"{n:,}".replace(",", "{,}")] + [f"{s.loc[n, m]:.1f}" for m in meths] for n in s.index]
parts.append(tab("Scalability of the DR step on EMNIST-Digits: total time [s] to compute the reduced representation "
                 "of $n$ samples (fit + transform for learned methods).", "tab:scalability",
                 ["\\textbf{$n$}"] + [f"\\textbf{{{SHORT.get(m, m)}}}" for m in meths], rows, "r|" + "c" * len(meths)))

# ---- DFFE vs resolution
res = pd.read_csv(R / "dffe_resolution.csv")
rows = [[f"${int(r.side_px)}\\times{int(r.side_px)}$", f"{int(r.pixels)}", f"{r.per_image_ms:.3f}"] for r in res.itertuples()]
parts.append(tab("DFFE extraction time per image as a function of image resolution.", "tab:resolution",
                 ["\\textbf{Resolution}", "\\textbf{Pixels}", "\\textbf{Time per image [ms]}"], rows, "c|cc"))

# ---- EMNIST rerun rows
e = pd.read_csv(R / "emnist_digits_rerun.csv").groupby(["classifier_name", "reduction_name"]).accuracy.agg(["mean", "std"]) * 100
ec = ["No Reduction", "DFFE (Flood Fill)", "LDA", "PCA", "TruncatedSVD", "PaCMAP", "UMAP"]
lines = []
for clf in ["CatBoost", "LightGBM"]:
    lines.append(f"{clf:22s} & " + " & ".join(f"${e.loc[(clf, r)]['mean']:.2f} \\pm {e.loc[(clf, r)]['std']:.2f}$" for r in ec) + r" \\")
(OUT / "emnist_rerun_rows.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

(OUT / "revision_tables.tex").write_text("\n".join(parts), encoding="utf-8")
print("\n".join(parts))
print("EMNIST rows:\n" + "\n".join(lines))
