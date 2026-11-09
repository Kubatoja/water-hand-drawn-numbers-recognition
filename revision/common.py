"""Shared helpers for the revision experiments (reviewer round 1).

Reuses the original pipeline (binarisation, DFFE, reducers from VectorManager)
and the original CV protocol: StratifiedKFold(5, shuffle=True, random_state=42).
Classifier hyper-parameters are copied verbatim from main.py (CLASSIFIER_REGISTRY).
"""
import gzip
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.model_selection import StratifiedKFold  # noqa: E402

from Testers.Shared.configs import DimensionalityReductionAlgorithm as DR, FloodConfig  # noqa: E402
from Testers.Shared.DataLoader import DataLoader, DataType  # noqa: E402
from Testers.Shared.MetricsCalculator import MetricsCalculator  # noqa: E402
from Testers.Shared.VectorManager import VectorManager  # noqa: E402

PIXEL_THRESHOLD = 0.2285805064971576
NUM_SEGMENTS = 7
N_COMPONENTS = 43
CV_SEED = 42
N_FOLDS = 5

DATA = ROOT / "Data"


# ---------------------------------------------------------------- datasets
def _read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        raw = f.read()
    ndim = raw[3]
    dims = [int.from_bytes(raw[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndim).reshape(dims)


def load_dataset(name: str):
    """Returns (pixels[N, H*W] in [0,1], labels[N], image_size, class_count).

    Only the training partition is loaded: the reported CV is computed on it.
    Row order follows the original distribution files, so CV folds match the
    original runs (EMNIST images are kept in their raw, transposed orientation,
    exactly as in the Kaggle CSV export used originally).
    """
    if name == "MNIST":
        x = _read_idx(DATA / "dl_mnist/train-images-idx3-ubyte.gz")
        y = _read_idx(DATA / "dl_mnist/train-labels-idx1-ubyte.gz")
        size = 28
    elif name == "Fashion-MNIST":
        x = _read_idx(DATA / "dl_fashion/train-images-idx3-ubyte.gz")
        y = _read_idx(DATA / "dl_fashion/train-labels-idx1-ubyte.gz")
        size = 28
    elif name == "EMNIST-Digits":
        x = _read_idx(DATA / "dl_emnist/gzip/emnist-digits-train-images-idx3-ubyte.gz")
        y = _read_idx(DATA / "dl_emnist/gzip/emnist-digits-train-labels-idx1-ubyte.gz")
        size = 28
    elif name == "USPS":
        raw = DataLoader(str(ROOT))._load_libsvm(str(DATA / "Usps/usps.bz2"))
        pix = np.asarray([r.pixels for r in raw], dtype=np.float64)
        lab = np.asarray([r.label for r in raw], dtype=np.int64)
        return pix, lab, 16, 10
    else:
        raise ValueError(name)
    pix = x.reshape(len(x), -1).astype(np.float64) / 255.0
    return pix, y.astype(np.int64), size, 10


# ---------------------------------------------------------------- features
REDUCTIONS = {
    "No Reduction": DR.NONE,
    "DFFE (Flood Fill)": DR.FLOOD_FILL,
    "LDA": DR.LDA,
    "PCA": DR.PCA,
    "TruncatedSVD": DR.TSVD,
    "PaCMAP": DR.PACMAP,
    "UMAP": DR.UMAP,
}


def make_config(reduction: str, image_size: int, class_count: int, num_segments: int = NUM_SEGMENTS):
    alg = REDUCTIONS[reduction]
    n_comp = min(N_COMPONENTS, class_count - 1) if alg == DR.LDA else N_COMPONENTS
    return SimpleNamespace(
        pixel_normalization_rate=PIXEL_THRESHOLD,
        dimensionality_reduction_algorithm=alg,
        dimensionality_reduction_n_components=n_comp,
        num_segments=num_segments,
        flood_config=FloodConfig.from_string("1111"),
        image_size=image_size,
        training_set_limit=99_999_999,
        class_count=class_count,
    )


def build_features(pixels, labels, reduction, image_size, class_count):
    """Same code path as the original runs (VectorManager._prepare_vectors_batch logic)."""
    cfg = make_config(reduction, image_size, class_count)
    vm = VectorManager(default_vectors_file=str(ROOT / "results/revision/_unused.csv"))
    binar = np.where(pixels > cfg.pixel_normalization_rate, 1, 0).reshape(-1, image_size, image_size)
    t0 = time.perf_counter()
    if cfg.dimensionality_reduction_algorithm == DR.NONE:
        X = binar.reshape(binar.shape[0], -1).astype(np.float64)
    else:
        X = vm.apply_dimensionality_reduction(
            binar, pixels.reshape(-1, image_size, image_size), labels, cfg, is_training_data=True)
    return np.asarray(X, dtype=np.float64), time.perf_counter() - t0


# ---------------------------------------------------------------- classifiers (params = main.py)
# With the default min_sum_hessian_in_leaf=1e-3, LightGBM diverges on EMNIST-Digits (training
# multi-logloss rises from 0.06 at iter 200 to 1.96 at iter 500: near-zero hessians -> exploding
# leaf values). A larger hessian floor removes this; used for the EMNIST-Digits re-run.
LGBM_STABLE = {"min_sum_hessian_in_leaf": 1.0}


def make_classifier(name: str, num_classes: int, lgbm_stable: bool = False):
    if name == "KNN":
        from sklearn.neighbors import KNeighborsClassifier
        return KNeighborsClassifier(n_neighbors=3, weights="uniform", algorithm="auto",
                                    leaf_size=30, p=2, metric="minkowski", n_jobs=-1)
    if name == "SVM":
        from sklearn.svm import SVC
        return SVC(C=1.0, kernel="poly", degree=9, gamma="scale", coef0=0.0,
                   shrinking=True, probability=False, random_state=42)
    if name == "MLP":
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier(hidden_layer_sizes=(800,), activation="relu", solver="adam",
                             alpha=0.0001, learning_rate="adaptive", learning_rate_init=0.001,
                             max_iter=200, random_state=42)
    if name == "XGBoost":
        from xgboost import XGBClassifier
        return XGBClassifier(learning_rate=0.1237, n_estimators=600, max_depth=4,
                             min_child_weight=1.0, gamma=0.0597, subsample=0.6455,
                             colsample_bytree=0.5871, reg_lambda=1.0, reg_alpha=0.0,
                             random_state=42, n_jobs=-1)
    if name == "CatBoost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(learning_rate=0.1, iterations=500, depth=6, subsample=0.8,
                                  colsample_bylevel=0.8, bootstrap_type="Bernoulli",
                                  random_seed=42, verbose=0, thread_count=-1,
                                  loss_function="MultiClass")
    if name == "LightGBM":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(learning_rate=0.1, n_estimators=500, max_depth=7, num_leaves=31,
                              subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
                              objective="multiclass", num_class=num_classes, verbosity=-1,
                              **(LGBM_STABLE if lgbm_stable else {}))
    raise ValueError(name)


def folds(y):
    return StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED).split(np.zeros(len(y)), y)


def evaluate_fold(clf_name, X_tr, y_tr, X_va, y_va, num_classes, lgbm_stable=False):
    model = make_classifier(clf_name, num_classes, lgbm_stable)
    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    train_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    y_pred = model.predict(X_va)
    test_time = time.perf_counter() - t0
    mc = MetricsCalculator()
    actual, pred = mc.prepare_labels(y_va, y_pred)
    m = mc.calculate_all_metrics(actual, pred, num_classes)
    correct = int(np.sum(actual == pred))
    return {
        "train_set_size": len(y_tr), "test_set_size": len(y_va),
        "correct_predictions": correct, "incorrect_predictions": len(y_va) - correct,
        "accuracy": correct / len(y_va), "precision": m["precision"], "recall": m["recall"],
        "f1_score": m["f1_score"], "training_time": train_time, "testing_time": test_time,
    }
