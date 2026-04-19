"""Config-driven experiment runner — manual mode.

Runs exactly the 162 experiments that are missing or broken
based on previous results analysis.

Classifiers in scope: KNN, SVM, MLP, XGBoost, LightGBM, CatBoost
Excluded (failed/GPU issues): TabICL, TabR, HyperFast
"""

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Dict, List

from Testers.Shared.configs import (
    DimensionalityReductionAlgorithm,
    FloodConfig,
    TestRunnerConfig,
)
from Testers.Shared.TestResultCollector import TestResultCollector
from Testers.Shared.dataset_config import (
    ARABIC_DATASET,
    EMNIST_BALANCED_DATASET,
    EMNIST_DIGITS_DATASET,
    FASHION_MNIST_DATASET,
    MNIST_DATASET,
    USPS_DATASET,
)

from Testers.CatBoostTester.CatBoostTestRunner import CatBoostTestRunner
from Testers.CatBoostTester.configs import CatBoostTestConfig
from Testers.LightGBMTester.LightGBMTestRunner import LightGBMTestRunner
from Testers.LightGBMTester.configs import LightGBMTestConfig
from Testers.KNNTester.KNNTestRunner import KNNTestRunner
from Testers.KNNTester.configs import KNNTestConfig
from Testers.SVMTester.SVMTestRunner import SVMTestRunner
from Testers.SVMTester.configs import SVMTestConfig
from Testers.MLPTester.MLPTestRunner import MLPTestRunner
from Testers.MLPTester.configs import MLPTestConfig
from Testers.XgBoostTester.XGBTestRunner import XGBTestRunner
from Testers.XgBoostTester.configs import XGBTestConfig


@dataclass(frozen=True)
class ClassifierSpec:
    key: str
    name: str
    runner_cls: Any
    config_cls: Any
    default_params: Dict[str, Any]


@dataclass(frozen=True)
class ReductionSpec:
    key: str
    name: str
    algorithm: DimensionalityReductionAlgorithm
    n_components: int
    training_set_limit: int = 99_999_999
    num_segments: int = 7
    pixel_normalization_rate: float = 0.2285805064971576
    flood_config: FloodConfig = field(default_factory=lambda: FloodConfig.from_string("1111"))


@dataclass(frozen=True)
class ExperimentSpec:
    dataset_key: str
    reduction_key: str
    classifier_key: str


# ============================================================================
# 1) CLASSIFIERS
# ============================================================================

CLASSIFIER_REGISTRY: Dict[str, ClassifierSpec] = {
    "KNN": ClassifierSpec(
        key="KNN",
        name="KNN",
        runner_cls=KNNTestRunner,
        config_cls=KNNTestConfig,
        default_params={
            "n_neighbors": 3,
            "weights": "uniform",
            "algorithm": "auto",
            "leaf_size": 30,
            "p": 2,
            "metric": "minkowski",
        },
    ),
    "SVM": ClassifierSpec(
        key="SVM",
        name="SVM",
        runner_cls=SVMTestRunner,
        config_cls=SVMTestConfig,
        default_params={
            "C": 1.0,
            "kernel": "poly",
            "degree": 9,
            "gamma": "scale",
            "coef0": 0.0,
            "shrinking": True,
            "probability": False,
        },
    ),
    "MLP": ClassifierSpec(
        key="MLP",
        name="MLP",
        runner_cls=MLPTestRunner,
        config_cls=MLPTestConfig,
        default_params={
            "hidden_layer_sizes": (800,),
            "activation": "relu",
            "solver": "adam",
            "alpha": 0.0001,
            "learning_rate": "adaptive",
            "learning_rate_init": 0.001,
            "max_iter": 200,
        },
    ),
    "XGBOOST": ClassifierSpec(
        key="XGBOOST",
        name="XGBoost",
        runner_cls=XGBTestRunner,
        config_cls=XGBTestConfig,
        default_params={
            "learning_rate": 0.1237,
            "n_estimators": 600,
            "max_depth": 4,
            "min_child_weight": 1.0,
            "gamma": 0.0597,
            "subsample": 0.6455,
            "colsample_bytree": 0.5871,
            "reg_lambda": 1.0,
            "reg_alpha": 0.0,
        },
    ),
    "CATBOOST": ClassifierSpec(
        key="CATBOOST",
        name="CatBoost",
        runner_cls=CatBoostTestRunner,
        config_cls=CatBoostTestConfig,
        default_params={
            "learning_rate": 0.1,
            "n_estimators": 500,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bylevel": 0.8,
            "random_state": 42,
            "verbose": 0,
            "thread_count": -1,
        },
    ),
    "LIGHTGBM": ClassifierSpec(
        key="LIGHTGBM",
        name="LightGBM",
        runner_cls=LightGBMTestRunner,
        config_cls=LightGBMTestConfig,
        default_params={
            "learning_rate": 0.1,
            "n_estimators": 500,
            "max_depth": 7,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
        },
    ),
}


# ============================================================================
# 2) DATASETS
# ============================================================================

DATASET_REGISTRY: Dict[str, Any] = {
    "MNIST": MNIST_DATASET,
    "FASHION_MNIST": FASHION_MNIST_DATASET,
    "EMNIST_DIGITS": EMNIST_DIGITS_DATASET,
    "EMNIST_BALANCED": EMNIST_BALANCED_DATASET,
    "ARABIC": ARABIC_DATASET,
    "USPS": USPS_DATASET,
}


# ============================================================================
# 3) REDUCTION METHODS
# ============================================================================

REDUCTION_REGISTRY: Dict[str, ReductionSpec] = {
    "NONE": ReductionSpec(
        key="NONE",
        name="No Reduction",
        algorithm=DimensionalityReductionAlgorithm.NONE,
        n_components=1,
    ),
    "DFFE": ReductionSpec(
        key="DFFE",
        name="DFFE (Flood Fill)",
        algorithm=DimensionalityReductionAlgorithm.FLOOD_FILL,
        n_components=43,
        num_segments=7,
        flood_config=FloodConfig.from_string("1111"),
    ),
    "PCA": ReductionSpec(
        key="PCA",
        name="PCA",
        algorithm=DimensionalityReductionAlgorithm.PCA,
        n_components=43,
    ),
    "LDA": ReductionSpec(
        key="LDA",
        name="LDA",
        algorithm=DimensionalityReductionAlgorithm.LDA,
        n_components=9,
    ),
    "ISOMAP": ReductionSpec(
        key="ISOMAP",
        name="Isomap",
        algorithm=DimensionalityReductionAlgorithm.ISOMAP,
        n_components=43,
    ),
    "UMAP": ReductionSpec(
        key="UMAP",
        name="UMAP",
        algorithm=DimensionalityReductionAlgorithm.UMAP,
        n_components=43,
    ),
    "TSVD": ReductionSpec(
        key="TSVD",
        name="TruncatedSVD",
        algorithm=DimensionalityReductionAlgorithm.TSVD,
        n_components=43,
    ),
    "PACMAP": ReductionSpec(
        key="PACMAP",
        name="PaCMAP",
        algorithm=DimensionalityReductionAlgorithm.PACMAP,
        n_components=43,
    ),
}


# ============================================================================
# EXPERIMENTS TO RUN  (162 missing / broken experiments)
#
# Breakdown:
#   - Arabic (all 8 reductions × 6 clf):          48 exp  — dataset never run
#   - USPS   (all 8 reductions × 6 clf):          48 exp  — dataset never run
#   - Isomap (MNIST/Fashion/EMNIST-D/B × 6 clf):  24 exp  — reduction never run
#   - CatBoost remaining datasets (overflow fix):  27 exp  — broken accuracy
#   - EMNIST-Balanced PaCMAP/UMAP remaining:        9 exp  — session aborted
#   - LightGBM EMNIST suspicious re-runs:           6 exp  — suspicious accuracy
# ============================================================================

EXPERIMENTS: List[ExperimentSpec] = [
    # ── Arabic — all reductions ──────────────────────────────────────────────
    ExperimentSpec("ARABIC", "DFFE",   "CATBOOST"),
    ExperimentSpec("ARABIC", "DFFE",   "KNN"),
    ExperimentSpec("ARABIC", "DFFE",   "LIGHTGBM"),
    ExperimentSpec("ARABIC", "DFFE",   "MLP"),
    ExperimentSpec("ARABIC", "DFFE",   "SVM"),
    ExperimentSpec("ARABIC", "DFFE",   "XGBOOST"),
    ExperimentSpec("ARABIC", "ISOMAP", "CATBOOST"),
    ExperimentSpec("ARABIC", "ISOMAP", "KNN"),
    ExperimentSpec("ARABIC", "ISOMAP", "LIGHTGBM"),
    ExperimentSpec("ARABIC", "ISOMAP", "MLP"),
    ExperimentSpec("ARABIC", "ISOMAP", "SVM"),
    ExperimentSpec("ARABIC", "ISOMAP", "XGBOOST"),
    ExperimentSpec("ARABIC", "LDA",    "CATBOOST"),
    ExperimentSpec("ARABIC", "LDA",    "KNN"),
    ExperimentSpec("ARABIC", "LDA",    "LIGHTGBM"),
    ExperimentSpec("ARABIC", "LDA",    "MLP"),
    ExperimentSpec("ARABIC", "LDA",    "SVM"),
    ExperimentSpec("ARABIC", "LDA",    "XGBOOST"),
    ExperimentSpec("ARABIC", "NONE",   "CATBOOST"),
    ExperimentSpec("ARABIC", "NONE",   "KNN"),
    ExperimentSpec("ARABIC", "NONE",   "LIGHTGBM"),
    ExperimentSpec("ARABIC", "NONE",   "MLP"),
    ExperimentSpec("ARABIC", "NONE",   "SVM"),
    ExperimentSpec("ARABIC", "NONE",   "XGBOOST"),
    ExperimentSpec("ARABIC", "PCA",    "CATBOOST"),
    ExperimentSpec("ARABIC", "PCA",    "KNN"),
    ExperimentSpec("ARABIC", "PCA",    "LIGHTGBM"),
    ExperimentSpec("ARABIC", "PCA",    "MLP"),
    ExperimentSpec("ARABIC", "PCA",    "SVM"),
    ExperimentSpec("ARABIC", "PCA",    "XGBOOST"),
    ExperimentSpec("ARABIC", "PACMAP", "CATBOOST"),
    ExperimentSpec("ARABIC", "PACMAP", "KNN"),
    ExperimentSpec("ARABIC", "PACMAP", "LIGHTGBM"),
    ExperimentSpec("ARABIC", "PACMAP", "MLP"),
    ExperimentSpec("ARABIC", "PACMAP", "SVM"),
    ExperimentSpec("ARABIC", "PACMAP", "XGBOOST"),
    ExperimentSpec("ARABIC", "TSVD",   "CATBOOST"),
    ExperimentSpec("ARABIC", "TSVD",   "KNN"),
    ExperimentSpec("ARABIC", "TSVD",   "LIGHTGBM"),
    ExperimentSpec("ARABIC", "TSVD",   "MLP"),
    ExperimentSpec("ARABIC", "TSVD",   "SVM"),
    ExperimentSpec("ARABIC", "TSVD",   "XGBOOST"),
    ExperimentSpec("ARABIC", "UMAP",   "CATBOOST"),
    ExperimentSpec("ARABIC", "UMAP",   "KNN"),
    ExperimentSpec("ARABIC", "UMAP",   "LIGHTGBM"),
    ExperimentSpec("ARABIC", "UMAP",   "MLP"),
    ExperimentSpec("ARABIC", "UMAP",   "SVM"),
    ExperimentSpec("ARABIC", "UMAP",   "XGBOOST"),

    # ── USPS — all reductions ────────────────────────────────────────────────
    ExperimentSpec("USPS", "DFFE",   "CATBOOST"),
    ExperimentSpec("USPS", "DFFE",   "KNN"),
    ExperimentSpec("USPS", "DFFE",   "LIGHTGBM"),
    ExperimentSpec("USPS", "DFFE",   "MLP"),
    ExperimentSpec("USPS", "DFFE",   "SVM"),
    ExperimentSpec("USPS", "DFFE",   "XGBOOST"),
    ExperimentSpec("USPS", "ISOMAP", "CATBOOST"),
    ExperimentSpec("USPS", "ISOMAP", "KNN"),
    ExperimentSpec("USPS", "ISOMAP", "LIGHTGBM"),
    ExperimentSpec("USPS", "ISOMAP", "MLP"),
    ExperimentSpec("USPS", "ISOMAP", "SVM"),
    ExperimentSpec("USPS", "ISOMAP", "XGBOOST"),
    ExperimentSpec("USPS", "LDA",    "CATBOOST"),
    ExperimentSpec("USPS", "LDA",    "KNN"),
    ExperimentSpec("USPS", "LDA",    "LIGHTGBM"),
    ExperimentSpec("USPS", "LDA",    "MLP"),
    ExperimentSpec("USPS", "LDA",    "SVM"),
    ExperimentSpec("USPS", "LDA",    "XGBOOST"),
    ExperimentSpec("USPS", "NONE",   "CATBOOST"),
    ExperimentSpec("USPS", "NONE",   "KNN"),
    ExperimentSpec("USPS", "NONE",   "LIGHTGBM"),
    ExperimentSpec("USPS", "NONE",   "MLP"),
    ExperimentSpec("USPS", "NONE",   "SVM"),
    ExperimentSpec("USPS", "NONE",   "XGBOOST"),
    ExperimentSpec("USPS", "PCA",    "CATBOOST"),
    ExperimentSpec("USPS", "PCA",    "KNN"),
    ExperimentSpec("USPS", "PCA",    "LIGHTGBM"),
    ExperimentSpec("USPS", "PCA",    "MLP"),
    ExperimentSpec("USPS", "PCA",    "SVM"),
    ExperimentSpec("USPS", "PCA",    "XGBOOST"),
    ExperimentSpec("USPS", "PACMAP", "CATBOOST"),
    ExperimentSpec("USPS", "PACMAP", "KNN"),
    ExperimentSpec("USPS", "PACMAP", "LIGHTGBM"),
    ExperimentSpec("USPS", "PACMAP", "MLP"),
    ExperimentSpec("USPS", "PACMAP", "SVM"),
    ExperimentSpec("USPS", "PACMAP", "XGBOOST"),
    ExperimentSpec("USPS", "TSVD",   "CATBOOST"),
    ExperimentSpec("USPS", "TSVD",   "KNN"),
    ExperimentSpec("USPS", "TSVD",   "LIGHTGBM"),
    ExperimentSpec("USPS", "TSVD",   "MLP"),
    ExperimentSpec("USPS", "TSVD",   "SVM"),
    ExperimentSpec("USPS", "TSVD",   "XGBOOST"),
    ExperimentSpec("USPS", "UMAP",   "CATBOOST"),
    ExperimentSpec("USPS", "UMAP",   "KNN"),
    ExperimentSpec("USPS", "UMAP",   "LIGHTGBM"),
    ExperimentSpec("USPS", "UMAP",   "MLP"),
    ExperimentSpec("USPS", "UMAP",   "SVM"),
    ExperimentSpec("USPS", "UMAP",   "XGBOOST"),

    # ── Isomap — missing for 4 existing datasets ─────────────────────────────
    ExperimentSpec("MNIST",          "ISOMAP", "CATBOOST"),
    ExperimentSpec("MNIST",          "ISOMAP", "KNN"),
    ExperimentSpec("MNIST",          "ISOMAP", "LIGHTGBM"),
    ExperimentSpec("MNIST",          "ISOMAP", "MLP"),
    ExperimentSpec("MNIST",          "ISOMAP", "SVM"),
    ExperimentSpec("MNIST",          "ISOMAP", "XGBOOST"),
    ExperimentSpec("FASHION_MNIST",  "ISOMAP", "CATBOOST"),
    ExperimentSpec("FASHION_MNIST",  "ISOMAP", "KNN"),
    ExperimentSpec("FASHION_MNIST",  "ISOMAP", "LIGHTGBM"),
    ExperimentSpec("FASHION_MNIST",  "ISOMAP", "MLP"),
    ExperimentSpec("FASHION_MNIST",  "ISOMAP", "SVM"),
    ExperimentSpec("FASHION_MNIST",  "ISOMAP", "XGBOOST"),
    ExperimentSpec("EMNIST_DIGITS",  "ISOMAP", "CATBOOST"),
    ExperimentSpec("EMNIST_DIGITS",  "ISOMAP", "KNN"),
    ExperimentSpec("EMNIST_DIGITS",  "ISOMAP", "LIGHTGBM"),
    ExperimentSpec("EMNIST_DIGITS",  "ISOMAP", "MLP"),
    ExperimentSpec("EMNIST_DIGITS",  "ISOMAP", "SVM"),
    ExperimentSpec("EMNIST_DIGITS",  "ISOMAP", "XGBOOST"),
    ExperimentSpec("EMNIST_BALANCED","ISOMAP", "CATBOOST"),
    ExperimentSpec("EMNIST_BALANCED","ISOMAP", "KNN"),
    ExperimentSpec("EMNIST_BALANCED","ISOMAP", "LIGHTGBM"),
    ExperimentSpec("EMNIST_BALANCED","ISOMAP", "MLP"),
    ExperimentSpec("EMNIST_BALANCED","ISOMAP", "SVM"),
    ExperimentSpec("EMNIST_BALANCED","ISOMAP", "XGBOOST"),

    # ── CatBoost — broken (overflow) on 4 existing datasets ──────────────────
    ExperimentSpec("MNIST",          "NONE",   "CATBOOST"),
    ExperimentSpec("MNIST",          "DFFE",   "CATBOOST"),
    ExperimentSpec("MNIST",          "PCA",    "CATBOOST"),
    ExperimentSpec("MNIST",          "TSVD",   "CATBOOST"),
    ExperimentSpec("MNIST",          "LDA",    "CATBOOST"),
    ExperimentSpec("MNIST",          "UMAP",   "CATBOOST"),
    ExperimentSpec("MNIST",          "PACMAP", "CATBOOST"),
    ExperimentSpec("FASHION_MNIST",  "NONE",   "CATBOOST"),
    ExperimentSpec("FASHION_MNIST",  "DFFE",   "CATBOOST"),
    ExperimentSpec("FASHION_MNIST",  "PCA",    "CATBOOST"),
    ExperimentSpec("FASHION_MNIST",  "TSVD",   "CATBOOST"),
    ExperimentSpec("FASHION_MNIST",  "LDA",    "CATBOOST"),
    ExperimentSpec("FASHION_MNIST",  "UMAP",   "CATBOOST"),
    ExperimentSpec("FASHION_MNIST",  "PACMAP", "CATBOOST"),
    ExperimentSpec("EMNIST_DIGITS",  "NONE",   "CATBOOST"),
    ExperimentSpec("EMNIST_DIGITS",  "DFFE",   "CATBOOST"),
    ExperimentSpec("EMNIST_DIGITS",  "PCA",    "CATBOOST"),
    ExperimentSpec("EMNIST_DIGITS",  "TSVD",   "CATBOOST"),
    ExperimentSpec("EMNIST_DIGITS",  "LDA",    "CATBOOST"),
    ExperimentSpec("EMNIST_DIGITS",  "UMAP",   "CATBOOST"),
    ExperimentSpec("EMNIST_DIGITS",  "PACMAP", "CATBOOST"),
    ExperimentSpec("EMNIST_BALANCED","NONE",   "CATBOOST"),
    ExperimentSpec("EMNIST_BALANCED","DFFE",   "CATBOOST"),
    ExperimentSpec("EMNIST_BALANCED","PCA",    "CATBOOST"),
    ExperimentSpec("EMNIST_BALANCED","TSVD",   "CATBOOST"),
    ExperimentSpec("EMNIST_BALANCED","LDA",    "CATBOOST"),
    ExperimentSpec("EMNIST_BALANCED","UMAP",   "CATBOOST"),
    ExperimentSpec("EMNIST_BALANCED","PACMAP", "CATBOOST"),

    # ── EMNIST-Balanced — aborted session (PaCMAP + UMAP) ────────────────────
    ExperimentSpec("EMNIST_BALANCED","PACMAP", "KNN"),
    ExperimentSpec("EMNIST_BALANCED","PACMAP", "LIGHTGBM"),
    ExperimentSpec("EMNIST_BALANCED","PACMAP", "MLP"),
    ExperimentSpec("EMNIST_BALANCED","PACMAP", "SVM"),
    ExperimentSpec("EMNIST_BALANCED","PACMAP", "XGBOOST"),
    ExperimentSpec("EMNIST_BALANCED","UMAP",   "LIGHTGBM"),
    ExperimentSpec("EMNIST_BALANCED","UMAP",   "MLP"),
    ExperimentSpec("EMNIST_BALANCED","UMAP",   "SVM"),
    ExperimentSpec("EMNIST_BALANCED","UMAP",   "XGBOOST"),

    # ── LightGBM — suspicious low accuracy, re-run ───────────────────────────
    ExperimentSpec("EMNIST_BALANCED","NONE",   "LIGHTGBM"),
    ExperimentSpec("EMNIST_BALANCED","DFFE",   "LIGHTGBM"),
    ExperimentSpec("EMNIST_BALANCED","LDA",    "LIGHTGBM"),
    ExperimentSpec("EMNIST_DIGITS",  "NONE",   "LIGHTGBM"),
    ExperimentSpec("EMNIST_DIGITS",  "UMAP",   "LIGHTGBM"),
]


class ExperimentRunner:
    def __init__(self) -> None:
        self.collector = TestResultCollector(algorithm_name="Main_Experiments")
        self.runner_config = TestRunnerConfig(
            force_regenerate_vectors=True,
            save_results_after_each_test=True,
            use_cross_validation=True,
            cv_n_folds=5,
        )
        self.summary_rows: List[Dict[str, Any]] = []

    def run(self, experiments: List[ExperimentSpec]) -> None:
        for idx, exp in enumerate(experiments, start=1):
            dataset    = DATASET_REGISTRY[exp.dataset_key]
            reduction  = REDUCTION_REGISTRY[exp.reduction_key]
            classifier = CLASSIFIER_REGISTRY[exp.classifier_key]

            print("-" * 100)
            print(
                f"[{idx}/{len(experiments)}] "
                f"dataset={dataset.display_name} | "
                f"reduction={reduction.name} | "
                f"classifier={classifier.name}"
            )

            try:
                test_config = self._build_test_config(classifier, reduction, dataset)
                runner = classifier.runner_cls(
                    train_dataset_path=dataset.train_path,
                    test_dataset_path=dataset.test_path,
                    train_data_type=dataset.data_type,
                    test_data_type=dataset.data_type,
                    train_labels_path=dataset.train_labels_path,
                    test_labels_path=dataset.test_labels_path,
                    config=self.runner_config,
                    external_collector=self.collector,
                )

                before = len(self.collector.results)
                start  = perf_counter()
                runner.run_tests([test_config])
                elapsed = perf_counter() - start

                if len(self.collector.results) > before:
                    result = self.collector.results[-1]
                    self._record(dataset.display_name, reduction.name, classifier.name,
                                 result.accuracy, elapsed, "success")
                    print(f"OK  accuracy={result.accuracy:.4f}  time={elapsed:.2f}s")
                else:
                    self._record(dataset.display_name, reduction.name, classifier.name,
                                 0.0, elapsed, "failed")
                    print(f"FAILED  time={elapsed:.2f}s")

            except Exception as exc:
                self._record(dataset.display_name, reduction.name, classifier.name,
                             0.0, 0.0, f"error: {exc}")
                print(f"ERROR: {exc}")

        self._print_summary()
        results_dir = self.collector.get_results_directory()
        if results_dir:
            print(f"Results saved in: {results_dir}")

    @staticmethod
    def _build_test_config(classifier: ClassifierSpec, reduction: ReductionSpec, dataset: Any) -> Any:
        params = dict(classifier.default_params)
        params.update({
            "class_count":   dataset.class_count,
            "image_size":    dataset.image_size,
            "dimensionality_reduction_algorithm":   reduction.algorithm,
            "dimensionality_reduction_n_components":
                min(reduction.n_components, dataset.class_count - 1)
                if reduction.algorithm == DimensionalityReductionAlgorithm.LDA
                else reduction.n_components,
            "training_set_limit":       reduction.training_set_limit,
            "pixel_normalization_rate": reduction.pixel_normalization_rate,
            "num_segments":             reduction.num_segments,
            "flood_config":             reduction.flood_config,
        })
        cfg = classifier.config_cls(**params)
        cfg.dataset_name   = dataset.display_name
        cfg.classifier_name = classifier.name
        cfg.reduction_name  = reduction.name
        return cfg

    def _record(self, dataset: str, reduction: str, classifier: str,
                accuracy: float, elapsed: float, status: str) -> None:
        self.summary_rows.append({
            "dataset": dataset, "reduction": reduction, "classifier": classifier,
            "accuracy": accuracy, "total_time": elapsed, "status": status,
        })

    def _print_summary(self) -> None:
        print("=" * 100)
        print("FINAL SUMMARY")
        print("=" * 100)
        for row in self.summary_rows:
            print(
                f"{row['dataset']:<22} | {row['reduction']:<20} | {row['classifier']:<12} | "
                f"acc={row['accuracy']:.4f} | time={row['total_time']:.2f}s | {row['status']}"
            )


def main() -> None:
    print(f"Experiments to run: {len(EXPERIMENTS)}")
    runner = ExperimentRunner()
    runner.run(EXPERIMENTS)


if __name__ == "__main__":
    main()