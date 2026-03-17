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
            "device": "auto",
        },
    ),
    "TABR": ClassifierSpec(
        key="TABR",
        name="TabR",
        runner_cls=TabRTestRunner,
        config_cls=TabRTestConfig,
        default_params={
            "n_epochs": 10,
            "batch_size": 128,
            "learning_rate": 0.01,
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
            "device": "auto",
        },
    ),
    "GRANDE": ClassifierSpec(
        key="GRANDE",
        name="Grande",
        runner_cls=GRANDETestRunner,
        config_cls=GRANDETestConfig,
        default_params={
            "n_estimators": 16,
            "max_depth": 8,
            "learning_rate": 0.01,
        },
    ),
}

# Choose which classifiers are active.
SELECTED_CLASSIFIERS: List[str] = ["GRANDE"]


# ============================================================================
# 2) DATASETS TO RUN
# ============================================================================

DATASET_REGISTRY: Dict[str, Any] = {
    "MNIST": MNIST_DATASET,
    "USPS": USPS_DATASET,
}

# Choose which datasets are active.
SELECTED_DATASETS: List[str] = ["USPS"]


# ============================================================================
# 3) REDUCTION METHODS TO RUN
# ============================================================================

REDUCTION_REGISTRY: Dict[str, ReductionSpec] = {
    "DFFE": ReductionSpec(
        key="DFFE",
        name="DFFE (Flood Fill)",
        algorithm=DimensionalityReductionAlgorithm.FLOOD_FILL,
        n_components=43,
        num_segments=7,
    SELECTED_CLASSIFIERS: List[str] = ["HYPERFAST", "TABR", "TABICL", "GRANDE"]
        flood_config=FloodConfig.from_string("1111"),
    ),
}

# Choose which reductions are active.
SELECTED_REDUCTIONS: List[str] = ["DFFE"]


# ============================================================================
# EXECUTION MODE
# ============================================================================

# "cartesian" -> every selected dataset x reduction x classifier
# "manual"    -> run only explicit MANUAL_EXPERIMENTS list
RUN_MODE = "manual"

# Manual list / array of experiments.
# USPS + DFFE + (all classifiers)
MANUAL_EXPERIMENTS: List[ExperimentSpec] = [
    ExperimentSpec("USPS", "DFFE", "HYPERFAST"),
    ExperimentSpec("USPS", "DFFE", "TABR"),
    ExperimentSpec("USPS", "DFFE", "TABICL"),
    ExperimentSpec("USPS", "DFFE", "GRANDE"),
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
        cfg = classifier.config_cls(
            **classifier.default_params,
            class_count=dataset.class_count,
            image_size=dataset.image_size,
            dimensionality_reduction_algorithm=reduction.algorithm,
            dimensionality_reduction_n_components=min(reduction.n_components, dataset.class_count - 1)
            if reduction.algorithm == DimensionalityReductionAlgorithm.LDA
            else reduction.n_components,
            training_set_limit=reduction.training_set_limit,
            pixel_normalization_rate=reduction.pixel_normalization_rate,
            num_segments=reduction.num_segments,
            flood_config=reduction.flood_config,
        )
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
