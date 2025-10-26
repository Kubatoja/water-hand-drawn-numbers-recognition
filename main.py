"""
SZABLON: Pełne porównanie wszystkich metod redukcji wymiarów ze wszystkimi klasyfikatorami na wszystkich datasetach

Ten szablon pozwala na:
- Test wszystkich kombinacji algorytmów redukcji wymiarów i klasyfikatorów
- Porównanie efektywności różnych metod wektoryzacji i klasyfikacji
- Użycie wszystkich podstawowych datasetów (bez MNIST-C)
- Automatyczne wykrywanie dostępności BFS/numba dla FLOOD_FILL
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import time
import numpy as np

from Testers.Shared.configs import FloodConfig, DimensionalityReductionAlgorithm, ReductionConfig, TestRunnerConfig
from Testers.KNNTester.KNNTestRunner import KNNTestRunner
from Testers.KNNTester.configs import KNNTestConfig
from Testers.XgBoostTester.XGBTestRunner import XGBTestRunner
from Testers.XgBoostTester.configs import XGBTestConfig
from Testers.MLPTester.MLPTestRunner import MLPTestRunner
from Testers.MLPTester.configs import MLPTestConfig
from Testers.SVMTester.SVMTestRunner import SVMTestRunner
from Testers.SVMTester.configs import SVMTestConfig
from Testers.Shared.TestResultCollector import TestResultCollector
from Testers.Shared.DataLoader import DataType
from Testers.Shared.test_runner_factory import TestRunnerFactory
from Testers.Shared.models import TestResult
from Testers.Shared import (
    BASIC_DATASETS,
    ALL_MNIST_C_DATASETS
)
from Testers.Shared.dataset_config import ARABIC_DATASET, EMNIST_BALANCED_DATASET, MNIST_DATASET
from BFS.bfs import calculate_flooded_vector

class ClassifierType(Enum):
    """Typy klasyfikatorów"""
    KNN = "knn"
    XGBOOST = "xgboost"
    MLP = "mlp"
    SVM = "svm"


@dataclass
class ClassifierConfig:
    """Konfiguracja klasyfikatora z domyślnymi parametrami"""
    classifier_type: ClassifierType
    name: str
    default_params: Dict[str, Any]


# Domyślne konfiguracje dla klasyfikatorów
DEFAULT_CLASSIFIERS = [
    ClassifierConfig(
        classifier_type=ClassifierType.KNN,
        name="K-Nearest Neighbors",
        default_params={
            "n_neighbors": 5,
            "weights": "uniform",
            "algorithm": "auto",
            "leaf_size": 30,
            "p": 2,
            "metric": "minkowski",
        }
    ),
    ClassifierConfig(
        classifier_type=ClassifierType.XGBOOST,
        name="XGBoost",
        default_params={
            "learning_rate": 0.1,
            "n_estimators": 100,
            "max_depth": 6,
            "min_child_weight": 1,
            "gamma": 0,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1,
            "reg_alpha": 0,
        }
    ),
    ClassifierConfig(
        classifier_type=ClassifierType.MLP,
        name="Multi-Layer Perceptron",
        default_params={
            "hidden_layer_sizes": (256,128,),
            "activation": "relu",
            "solver": "adam",
            "alpha": 0.0001,
            "learning_rate": "constant",
            "learning_rate_init": 0.001,
            "max_iter": 1000,
        }
    ),
    ClassifierConfig(
        classifier_type=ClassifierType.SVM,
        name="Support Vector Machine",
        default_params={
            "C": 1.0,
            "kernel": "rbf",
            "degree": 3,
            "gamma": "scale",
            "coef0": 0.0,
            "shrinking": True,
            "probability": False,
        }
    ),
]


class FullComparisonRunner:
    """Klasa odpowiedzialna za pełne porównanie wszystkich metod redukcji wymiarów ze wszystkimi klasyfikatorami"""

    def __init__(self):
        self.result_collector = TestResultCollector(algorithm_name="Full_Comparison")
        self.global_test_index = 0
        self.all_results = []

    def run_comparison(self, datasets: List, classifiers: List[ClassifierConfig]) -> None:
        """Uruchamia pełne porównanie dla wszystkich datasetów i klasyfikatorów"""
        total_start_time = time.time()

        for dataset in datasets:
            print(f"\n{'='*100}")
            print(f"TESTOWANIE DATASETU: {dataset.display_name} ({dataset.class_count} klas, {dataset.image_size}x{dataset.image_size})")
            print(f"{'='*100}")

            reduction_configs = self.get_reduction_configs_for_dataset(dataset)
            self._run_dataset_comparison(dataset, reduction_configs, classifiers)

        total_time = time.time() - total_start_time
        self._print_final_summary(datasets, classifiers, total_time)

    def _run_dataset_comparison(self, dataset, reduction_configs: List[ReductionConfig], classifiers: List[ClassifierConfig]) -> None:
        """Uruchamia porównanie dla pojedynczego datasetu"""
        for reduction_config in reduction_configs:
            print(f"\n--- Testowanie metody redukcji: {reduction_config.name} ---")
            self._run_reduction_comparison(dataset, reduction_config, classifiers)

    def _run_reduction_comparison(self, dataset, reduction_config: ReductionConfig, classifiers: List[ClassifierConfig]) -> None:
        """Uruchamia porównanie dla pojedynczej metody redukcji"""
        # Najpierw wygeneruj wektory dla tej kombinacji dataset+redukcja (używając KNN)
        print(f"  Generowanie wektorów dla {reduction_config.name}...")
        first_classifier = classifiers[0]  # KNN jako pierwszy
        first_test_config = self.create_test_config(first_classifier, reduction_config, dataset)
        # Use display_name so MNIST-C entries include the corruption variant
        first_test_config.dataset_name = dataset.display_name
        first_test_config.classifier_name = first_classifier.name
        first_test_config.reduction_name = reduction_config.name

        first_runner = self.create_test_runner(first_classifier, dataset, self.result_collector)
        first_start_time = time.time()
        first_test_results = first_runner.run_tests([first_test_config])
        first_test_result = first_test_results[-1]
        first_end_time = time.time()

        # Zapisz wynik pierwszego testu
        self._save_test_result(first_test_result, dataset, first_classifier, reduction_config, first_end_time - first_start_time)
        print(f"    ✓ {first_classifier.name}: Accuracy = {first_test_result.accuracy:.4f} (time: {first_end_time - first_start_time:.2f}s)")

        # Teraz przetestuj pozostałe klasyfikatory na tych samych wektorach
        for classifier in classifiers[1:]:  # Pomiń pierwszego (KNN)
            print(f"  Test: {classifier.name} na {reduction_config.name}")
            self._run_single_classifier_test(classifier, reduction_config, dataset)

    def _run_single_classifier_test(self, classifier: ClassifierConfig, reduction_config: ReductionConfig, dataset) -> None:
        """Uruchamia test dla pojedynczego klasyfikatora"""
        print(f"Starting test for {classifier.name}")
        try:
            test_config = self.create_test_config(classifier, reduction_config, dataset)
            # Use display_name so MNIST-C entries include the corruption variant
            test_config.dataset_name = dataset.display_name
            test_config.classifier_name = classifier.name
            test_config.reduction_name = reduction_config.name

            runner = self.create_test_runner(classifier, dataset, self.result_collector)
            start_time = time.time()
            test_results = runner.run_tests([test_config])
            test_result = test_results[-1]
            end_time = time.time()

            self._save_test_result(test_result, dataset, classifier, reduction_config, end_time - start_time)
            print(f"    ✓ Accuracy = {test_result.accuracy:.4f} (time: {end_time - start_time:.2f}s)")

        except Exception as e:
            print(f"    ✗ Błąd for {classifier.name}: {str(e)}")
            self._save_failed_test(dataset, classifier, reduction_config, str(e))

    def _save_test_result(self, test_result, dataset, classifier: ClassifierConfig, reduction_config: ReductionConfig, total_time: float) -> None:
        """Zapisuje wynik testu do wspólnego collector'a i listy wyników"""
        print(f"Saving result for {classifier.name} with accuracy {test_result.accuracy}")
        self.result_collector.add_success_and_save(test_result, self.global_test_index)
        self.global_test_index += 1

        result_dict = {
            # Use display_name to reflect corruption variant for MNIST-C
            "dataset": dataset.display_name,
            "classifier": classifier.name,
            "reduction": reduction_config.name,
            "algorithm": reduction_config.algorithm.value if hasattr(reduction_config.algorithm, 'value') else str(reduction_config.algorithm),
            "n_components": test_result.actual_n_components or reduction_config.n_components,
            "training_set_limit": reduction_config.training_set_limit,
            "accuracy": test_result.accuracy,
            "training_time": test_result.training_time,
            "prediction_time": test_result.execution_time,
            "total_time": total_time,
            "status": "success",
        }
        self.all_results.append(result_dict)

    def _save_failed_test(self, dataset, classifier: ClassifierConfig, reduction_config: ReductionConfig, error: str) -> None:
        """Zapisuje nieudany test"""
        result_dict = {
            # Use display_name to reflect corruption variant for MNIST-C
            "dataset": dataset.display_name,
            "classifier": classifier.name,
            "reduction": reduction_config.name,
            "algorithm": reduction_config.algorithm.value if hasattr(reduction_config.algorithm, 'value') else str(reduction_config.algorithm),
            "n_components": reduction_config.n_components,  # Dla błędów używamy z konfiguracji
            "training_set_limit": reduction_config.training_set_limit,
            "accuracy": 0.0,
            "training_time": 0.0,
            "prediction_time": 0.0,
            "total_time": 0.0,
            "status": f"error: {error}",
        }
        self.all_results.append(result_dict)

    def get_reduction_configs_for_dataset(self, dataset) -> List[ReductionConfig]:
        """Generuje konfiguracje redukcji wymiarów dostosowane do datasetu"""
        full_dimension = dataset.image_size ** 2
        max_lda_components = dataset.class_count - 1

        base_training_limit = 99999999

        configs = [
            ReductionConfig(
                name="Brak redukcji",
                algorithm=DimensionalityReductionAlgorithm.NONE,
                n_components=full_dimension,
                training_set_limit=base_training_limit,
                requires_bfs=False,
            ),
            ReductionConfig(
                name="PCA",
                algorithm=DimensionalityReductionAlgorithm.PCA,
                n_components=43,  
                training_set_limit=base_training_limit,
                requires_bfs=False,
            ),
            ReductionConfig(
                name="LDA",
                algorithm=DimensionalityReductionAlgorithm.LDA,
                n_components=max_lda_components,
                training_set_limit=base_training_limit,
                requires_bfs=False,
            ),
            ReductionConfig(
                name="UMAP",
                algorithm=DimensionalityReductionAlgorithm.UMAP,
                n_components=43, 
                training_set_limit=base_training_limit,
                requires_bfs=False,
            ),
            ReductionConfig(
                name="Flood Fill",
                algorithm=DimensionalityReductionAlgorithm.FLOOD_FILL,
                n_components=43,
                num_segments=7,
                pixel_normalization_rate=0.2285805064971576,  # Optymalna wartość
                flood_config=FloodConfig.from_string("1111"),
                training_set_limit=base_training_limit,
                requires_bfs=True,
            )
        ]

        return configs

    def create_test_config(self, classifier: ClassifierConfig, reduction_config: ReductionConfig, dataset) -> Any:
        """Tworzy konfigurację testu dla danego klasyfikatora i redukcji"""
        base_params = {
            "class_count": dataset.class_count,
            "image_size": dataset.image_size,
            "dimensionality_reduction_algorithm": reduction_config.algorithm,
            "dimensionality_reduction_n_components": reduction_config.n_components,
            "training_set_limit": reduction_config.training_set_limit,
            "pixel_normalization_rate": reduction_config.pixel_normalization_rate,
        }

        if classifier.classifier_type == ClassifierType.KNN:
            return KNNTestConfig(
                **classifier.default_params,
                num_segments=reduction_config.num_segments,
                flood_config=reduction_config.flood_config,
                **base_params,
            )
        elif classifier.classifier_type == ClassifierType.XGBOOST:
            return XGBTestConfig(
                **classifier.default_params,
                num_segments=reduction_config.num_segments,
                flood_config=reduction_config.flood_config,
                **base_params,
            )
        elif classifier.classifier_type == ClassifierType.MLP:
            return MLPTestConfig(
                **classifier.default_params,
                num_segments=reduction_config.num_segments,
                flood_config=reduction_config.flood_config,
                **base_params,
            )
        elif classifier.classifier_type == ClassifierType.SVM:
            return SVMTestConfig(
                **classifier.default_params,
                num_segments=reduction_config.num_segments,
                flood_config=reduction_config.flood_config,
                **base_params,
            )
        else:
            raise ValueError(f"Nieznany typ klasyfikatora: {classifier.classifier_type}")

    def create_test_runner(self, classifier: ClassifierConfig, dataset, external_collector: TestResultCollector):
        """Tworzy odpowiedni test runner dla klasyfikatora używając fabryki"""
        config = TestRunnerConfig()  # Domyślna konfiguracja, bez zapisywania
        if classifier.classifier_type == ClassifierType.KNN:
            return KNNTestRunner(
                train_dataset_path=dataset.train_path,
                test_dataset_path=dataset.test_path,
                train_data_type=dataset.data_type,
                test_data_type=dataset.data_type,
                train_labels_path=dataset.train_labels_path,
                test_labels_path=dataset.test_labels_path,
                config=config,
                external_collector=external_collector
            )
        elif classifier.classifier_type == ClassifierType.XGBOOST:
            return XGBTestRunner(
                train_dataset_path=dataset.train_path,
                test_dataset_path=dataset.test_path,
                train_data_type=dataset.data_type,
                test_data_type=dataset.data_type,
                train_labels_path=dataset.train_labels_path,
                test_labels_path=dataset.test_labels_path,
                config=config,
                external_collector=external_collector
            )
        elif classifier.classifier_type == ClassifierType.MLP:
            return MLPTestRunner(
                train_dataset_path=dataset.train_path,
                test_dataset_path=dataset.test_path,
                train_data_type=dataset.data_type,
                test_data_type=dataset.data_type,
                train_labels_path=dataset.train_labels_path,
                test_labels_path=dataset.test_labels_path,
                config=config,
                external_collector=external_collector
            )
        elif classifier.classifier_type == ClassifierType.SVM:
            return SVMTestRunner(
                train_dataset_path=dataset.train_path,
                test_dataset_path=dataset.test_path,
                train_data_type=dataset.data_type,
                test_data_type=dataset.data_type,
                train_labels_path=dataset.train_labels_path,
                test_labels_path=dataset.test_labels_path,
                config=config,
                external_collector=external_collector
            )
        else:
            raise ValueError(f"Nieznany typ klasyfikatora: {classifier.classifier_type}")

    def _print_final_summary(self, datasets: List, classifiers: List[ClassifierConfig], total_time: float) -> None:
        """Wyświetla końcowe podsumowanie wyników"""
        print(f"\n{'='*120}")
        print("PODSUMOWANIE WYNIKÓW PORÓWNANIA")
        print(f"{'='*120}")
        print(f"Testowane datasety: {[d.display_name for d in datasets]}")
        print(f"Testowane klasyfikatory: {[c.name for c in classifiers]}")
        reduction_names = [c.name for c in self.get_reduction_configs_for_dataset(datasets[0])]
        print(f"Testowane metody redukcji: {reduction_names}")
        print(f"Całkowity czas wykonania: {total_time:.2f}s")

        # Filtruj tylko udane wyniki
        successful_results = [r for r in self.all_results if r["status"] == "success"]

        if successful_results:
            # Sortowanie po dokładności (malejąco)
            results_sorted = sorted(successful_results, key=lambda x: x["accuracy"], reverse=True)

            print(f"\n{'Lp.':<2} {'Dataset':<12} {'Klasyfikator':<20} {'Redukcja':<15} {'Accuracy':<10} {'Czas total':<12}")
            print("-" * 120)

            for i, result in enumerate(results_sorted[:20], 1):  # Pokaż top 20
                print(f"{i:<2} {result['dataset']:<12} {result['classifier']:<20} {result['reduction']:<15} {result['accuracy']:.4f} {result['total_time']:.2f}s")

            print(f"\n{'='*120}")

            # Najlepszy wynik
            best_result = results_sorted[0]
            print(f"🏆 NAJLEPSZY WYNIK: {best_result['dataset']} + {best_result['classifier']} + {best_result['reduction']}")
            print(f"   Accuracy: {best_result['accuracy']:.4f}")
            print(f"   Czas wykonania: {best_result['total_time']:.2f}s")

            # Podsumowanie per dataset
            print(f"\nPODSUMOWANIE PER DATASET:")
            for dataset in datasets:
                # Compare using display_name because saved results use display_name for MNIST-C
                dataset_results = [r for r in successful_results if r["dataset"] == dataset.display_name]
                if dataset_results:
                    avg_accuracy = sum(r["accuracy"] for r in dataset_results) / len(dataset_results)
                    best_for_dataset = max(dataset_results, key=lambda x: x["accuracy"])
                    print(f"  {dataset.display_name:<12}: Średnia accuracy = {avg_accuracy:.4f}, Najlepsza = {best_for_dataset['classifier']} + {best_for_dataset['reduction']} ({best_for_dataset['accuracy']:.4f})")

            # Podsumowanie per klasyfikator
            print(f"\nPODSUMOWANIE PER KLASYFIKATOR:")
            for classifier in classifiers:
                classifier_results = [r for r in successful_results if r["classifier"] == classifier.name]
                if classifier_results:
                    avg_accuracy = sum(r["accuracy"] for r in classifier_results) / len(classifier_results)
                    best_for_classifier = max(classifier_results, key=lambda x: x["accuracy"])
                    print(f"  {classifier.name:<20}: Średnia accuracy = {avg_accuracy:.4f}, Najlepsza = {best_for_classifier['dataset']} + {best_for_classifier['reduction']} ({best_for_classifier['accuracy']:.4f})")

        else:
            print("❌ Brak udanych testów!")

        # Pokaż również nieudane testy
        failed_results = [r for r in self.all_results if r["status"] != "success"]
        if failed_results:
            print(f"\n❌ NIEUDANE TESTY:")
            for result in failed_results[:10]:  # Pokaż pierwsze 10 błędów
                print(f"   - {result['dataset']} + {result['classifier']} + {result['reduction']}: {result['status']}")


def main():
    """
    ========================================================================
    PEŁNE PORÓWNANIE WSZYSTKICH METOD REDUKCJI WYMIARÓW ZE WSZYSTKIMI KLASYFIKATORAMI
    ========================================================================
    """

    # ========================================================================
    # 1. WYBÓR DATASETÓW
    # ========================================================================


    # Use only the Arabic dataset for testing
    datasets = ALL_MNIST_C_DATASETS  # Arabic dataset only

    classifiers_to_test = DEFAULT_CLASSIFIERS
    

    # ========================================================================
    # 2. URUCHOMIENIE TESTÓW DLA WSZYSTKICH DATASETÓW
    # ========================================================================

    runner = FullComparisonRunner()
    runner.run_comparison(datasets, classifiers_to_test)

    # ========================================================================
    # 3. ZAPIS WSZYSTKICH WYNIKÓW DO WSPÓLNEGO PLIKU
    # ========================================================================

    print(f"\n{'='*100}")
    print("ZAPIS WYNIKÓW DO WSPÓLNEGO PLIKU CSV")
    print(f"{'='*100}")

    try:
        # Wyniki są już zapisane na bieżąco
        print("✓ Wszystkie wyniki zostały zapisane na bieżąco do wspólnego pliku CSV")
    except Exception as e:
        print(f"✗ Błąd podczas zapisywania wyników: {str(e)}")


if __name__ == "__main__":
    main()