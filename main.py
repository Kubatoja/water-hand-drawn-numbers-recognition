"""Config-driven experiment runner.

Three central config sections:
1) classifiers
2) datasets
3) reduction methods

Execution modes:
- cartesian: every dataset x every reduction x every classifier
- manual: explicit list/array of experiment tuples
"""

from dataclasses import dataclass, field
from itertools import product
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

from Testers.TabICLTester.TabICLTestRunner import TabICLTestRunner
from Testers.TabICLTester.configs import TabICLTestConfig
from Testers.HyperFastTester.HyperFastTestRunner import HyperFastTestRunner
from Testers.HyperFastTester.configs import HyperFastTestConfig
from Testers.TabR.TabRTestRunner import TabRTestRunner
from Testers.TabR.configs import TabRTestConfig
from Testers.GrandeTester.Grandetestrunner import GRANDETestRunner
from Testers.GrandeTester.configs import GRANDETestConfig
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
# 1) CLASSIFIERS TO RUN
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
            "kernel": "poly",  # Zmieniono z rbf na poly zgodnie z wynikami dla MNIST
            "degree": 9,       # Virtual SVM z wielomianem 9. stopnia osiągnął błąd 0.56%
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
            "hidden_layer_sizes": (800,),  # 784-800-10 architecture
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
            "n_estimators": 600,     # Zmieniono z 100 na 600
            "max_depth": 4,          # Zmieniono z 6 na 4
            "min_child_weight": 1.0,
            "gamma": 0.0597,         # Zmieniono z 0.0 na 0.0597
            "subsample": 0.6455,     # Zmieniono z 0.8 na 0.6455
            "colsample_bytree": 0.5871, # Zmieniono z 0.8 na 0.5871
            "reg_lambda": 1.0,
            "reg_alpha": 0.0,
        },
    ),
    "TABICL": ClassifierSpec(
        key="TABICL",
        name="TabICL",
        runner_cls=TabICLTestRunner,
        config_cls=TabICLTestConfig,
        default_params={
            "n_estimators": 16,
            "softmax_temperature": 0.9,
            "outlier_threshold": 4.0,
            "device": "cuda",
            "random_state": 42,
        },
    ),
    "TABR": ClassifierSpec(
        key="TABR",
        name="TabR",
        runner_cls=TabRTestRunner,
        config_cls=TabRTestConfig,
        default_params={
            "n_epochs": 10,
            "batch_size": 256,
            "learning_rate": 0.001,
            "device": "cuda",
            "random_state": 42,
        },
    ),
    "GRANDE": ClassifierSpec(
        key="GRANDE",
        name="GRANDE",
        runner_cls=GRANDETestRunner,
        config_cls=GRANDETestConfig,
        default_params={
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": 42,
            "verbose": 0,
            "device": "cuda",
        },
    ),
    "HYPERFAST": ClassifierSpec(
        key="HYPERFAST",
        name="HyperFast",
        runner_cls=HyperFastTestRunner,
        config_cls=HyperFastTestConfig,
        default_params={
            "n_ensemble": 16,
            "batch_size": 2048,
            "nn_bias": 0.0,
            "optimization": "optimize",
            "optimize_steps": 64,
            "device": "cuda",
            "random_state": 42,
        },
    ),
}

# Choose which classifiers are active.
SELECTED_CLASSIFIERS: List[str] = ["KNN", "SVM", "MLP", "XGBOOST", "TABICL", "TABR", "GRANDE", "HYPERFAST"]


# ============================================================================
# 2) DATASETS TO RUN
# ============================================================================

DATASET_REGISTRY: Dict[str, Any] = {
    "MNIST": MNIST_DATASET,
    "FASHION_MNIST": FASHION_MNIST_DATASET,
    "EMNIST_DIGITS": EMNIST_DIGITS_DATASET,
    "EMNIST_BALANCED": EMNIST_BALANCED_DATASET,
    "ARABIC": ARABIC_DATASET,
    "USPS": USPS_DATASET,
}

# Choose which datasets are active.
SELECTED_DATASETS: List[str] = [
    "MNIST",
    "FASHION_MNIST",
    "EMNIST_DIGITS",
    "EMNIST_BALANCED",
    "ARABIC",
    "USPS",
]


# ============================================================================
# 3) REDUCTION METHODS TO RUN
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
}

# Choose which reductions are active.
SELECTED_REDUCTIONS: List[str] = ["NONE", "DFFE", "PCA", "LDA", "ISOMAP", "UMAP"]


# ============================================================================
# EXECUTION MODE
# ============================================================================

# "cartesian" -> every selected dataset x reduction x classifier
# "manual"    -> run only explicit MANUAL_EXPERIMENTS list
RUN_MODE = "manual"

# Manual list / array of experiments.
# Not used in cartesian mode, but useful for one-off debugging.
MANUAL_EXPERIMENTS: List[ExperimentSpec] = [
    ExperimentSpec("USPS", "DFFE", "KNN"),
    ExperimentSpec("USPS", "DFFE", "SVM"),
    ExperimentSpec("USPS", "DFFE", "MLP"),
    ExperimentSpec("USPS", "DFFE", "XGBOOST"),
    ExperimentSpec("USPS", "DFFE", "TABICL"),
    ExperimentSpec("USPS", "DFFE", "TABR"),
    ExperimentSpec("USPS", "DFFE", "GRANDE"),
    ExperimentSpec("USPS", "DFFE", "HYPERFAST"),
]


class ExperimentRunner:
    def __init__(self) -> None:
        self.collector = TestResultCollector(algorithm_name="Main_Experiments")
        self.runner_config = TestRunnerConfig(
            force_regenerate_vectors=False,
            save_results_after_each_test=True,
            use_cross_validation=True,
            cv_n_folds=5,
        )
        self.summary_rows: List[Dict[str, Any]] = []

    def run(self, experiments: List[ExperimentSpec]) -> None:
        for idx, exp in enumerate(experiments, start=1):
            dataset = DATASET_REGISTRY[exp.dataset_key]
            reduction = REDUCTION_REGISTRY[exp.reduction_key]
            classifier = CLASSIFIER_REGISTRY[exp.classifier_key]

            print("-" * 100)
            print(
                f"[{idx}/{len(experiments)}] "
                f"dataset={dataset.display_name} | reduction={reduction.name} | classifier={classifier.name}"
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

                before_success = len(self.collector.results)
                start = perf_counter()
                runner.run_tests([test_config])
                total_time = perf_counter() - start
                after_success = len(self.collector.results)

                if after_success > before_success:
                    result = self.collector.results[-1]
                    self._add_summary_row(
                        dataset.display_name,
                        reduction.name,
                        classifier.name,
                        result.accuracy,
                        total_time,
                        "success",
                    )
                    print(f"OK accuracy={result.accuracy:.4f} total_time={total_time:.2f}s")
                else:
                    self._add_summary_row(
                        dataset.display_name,
                        reduction.name,
                        classifier.name,
                        0.0,
                        total_time,
                        "failed",
                    )
                    print(f"FAILED total_time={total_time:.2f}s")

            except Exception as exc:
                self._add_summary_row(
                    dataset.display_name,
                    reduction.name,
                    classifier.name,
                    0.0,
                    0.0,
                    f"error: {exc}",
                )
                print(f"ERROR: {exc}")

        self._print_summary()
        results_dir = self.collector.get_results_directory()
        if results_dir:
            print(f"Results saved incrementally in: {results_dir}")

    @staticmethod
    def _build_test_config(classifier: ClassifierSpec, reduction: ReductionSpec, dataset: Any) -> Any:
        params = dict(classifier.default_params)
        params.update(
            {
                "class_count": dataset.class_count,
                "image_size": dataset.image_size,
                "dimensionality_reduction_algorithm": reduction.algorithm,
                "dimensionality_reduction_n_components":
                    min(reduction.n_components, dataset.class_count - 1)
                    if reduction.algorithm == DimensionalityReductionAlgorithm.LDA
                    else reduction.n_components,
                "training_set_limit": reduction.training_set_limit,
                "pixel_normalization_rate": reduction.pixel_normalization_rate,
                "num_segments": reduction.num_segments,
                "flood_config": reduction.flood_config,
            }
        )
        cfg = classifier.config_cls(**params)
        cfg.dataset_name = dataset.display_name
        cfg.classifier_name = classifier.name
        cfg.reduction_name = reduction.name
        return cfg

    def _add_summary_row(
        self,
        dataset_name: str,
        reduction_name: str,
        classifier_name: str,
        accuracy: float,
        total_time: float,
        status: str,
    ) -> None:
        self.summary_rows.append(
            {
                "dataset": dataset_name,
                "reduction": reduction_name,
                "classifier": classifier_name,
                "accuracy": accuracy,
                "total_time": total_time,
                "status": status,
            }
        )

    def _print_summary(self) -> None:
        print("=" * 100)
        print("FINAL SUMMARY")
        print("=" * 100)
        for row in self.summary_rows:
            print(
                f"{row['dataset']:<12} | {row['reduction']:<18} | {row['classifier']:<20} | "
                f"acc={row['accuracy']:.4f} | time={row['total_time']:.2f}s | {row['status']}"
            )


def build_experiments() -> List[ExperimentSpec]:
    if RUN_MODE == "cartesian":
        return [
            ExperimentSpec(dataset_key, reduction_key, classifier_key)
            for dataset_key, reduction_key, classifier_key in product(
                SELECTED_DATASETS,
                SELECTED_REDUCTIONS,
                SELECTED_CLASSIFIERS,
            )
        ]

    if RUN_MODE == "manual":
        return MANUAL_EXPERIMENTS

    raise ValueError(f"Unsupported RUN_MODE: {RUN_MODE}")


def validate_selection() -> None:
    for key in SELECTED_CLASSIFIERS:
        if key not in CLASSIFIER_REGISTRY:
            raise ValueError(f"Unknown classifier key: {key}")

    for key in SELECTED_DATASETS:
        if key not in DATASET_REGISTRY:
            raise ValueError(f"Unknown dataset key: {key}")

    for key in SELECTED_REDUCTIONS:
        if key not in REDUCTION_REGISTRY:
            raise ValueError(f"Unknown reduction key: {key}")

    for exp in MANUAL_EXPERIMENTS: 
        if exp.dataset_key not in DATASET_REGISTRY:
            raise ValueError(f"Unknown manual dataset key: {exp.dataset_key}")
        if exp.reduction_key not in REDUCTION_REGISTRY:
            raise ValueError(f"Unknown manual reduction key: {exp.reduction_key}")
        if exp.classifier_key not in CLASSIFIER_REGISTRY:
            raise ValueError(f"Unknown manual classifier key: {exp.classifier_key}")


def main() -> None:
    validate_selection()
    experiments = build_experiments()
    print(f"RUN_MODE={RUN_MODE}; experiments={len(experiments)}")

    runner = ExperimentRunner()
    runner.run(experiments)


if __name__ == "__main__":
    main()
