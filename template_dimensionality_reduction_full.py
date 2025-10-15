"""
SZABLON: Pełne porównanie wszystkich algorytmów redukcji wymiarów z XGBoost na MNIST

Ten szablon pozwala na:
- Test wszystkich algorytmów redukcji wymiarów z XGBoost
- Porównanie efektywności różnych metod wektoryzacji
- Użycie datasetu MNIST
- Automatyczne wykrywanie dostępności BFS/numba dla FLOOD_FILL
"""

from typing import List, Dict
from Testers.Shared.configs import FloodConfig, DimensionalityReductionAlgorithm
from Testers.XgBoostTester.XGBTestRunner import XGBTestRunner
from Testers.XgBoostTester.configs import XGBTestConfig
from Testers.Shared.DataLoader import DataType
from Testers.Shared import (
    EMNIST_BALANCED_DATASET,
    MNIST_DATASET
)

# Sprawdź dostępność BFS
try:
    from BFS.bfs import calculate_flooded_vector
    BFS_AVAILABLE = True
    print("✓ BFS/numba dostępny - FLOOD_FILL będzie testowany")
except ImportError:
    BFS_AVAILABLE = False
    print("⚠ BFS/numba niedostępny - FLOOD_FILL zostanie pominięty")


def main():
    """
    ========================================================================
    PEŁNE PORÓWNANIE ALGORYTMÓW REDUKCJI WYMIARÓW Z XGBoost NA MNIST
    ========================================================================
    """

    # ========================================================================
    # 1. WYBÓR ALGORYTMU
    # ========================================================================

    algorithm = "XGBoost"  # Stały wybór dla tego szablonu


    # ========================================================================
    # 2. WYBÓR DATASETU
    # ========================================================================

    dataset = MNIST_DATASET


    # ========================================================================
    # 3. DEFINICJA KONFIGURACJI REDUKCJI WYMIARÓW
    # ========================================================================

    # Bazowe parametry XGBoost (stałe dla wszystkich testów)
    base_xgb_params = {
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

    # Konfiguracje dla różnych algorytmów redukcji
    reduction_configs = [
        {
            "name": "Brak redukcji",
            "algorithm": DimensionalityReductionAlgorithm.NONE,
            "n_components": 784,  # Pełny wymiar MNIST
            "training_set_limit": 999999,  # Mniejszy zbiór dla szybszych testów
            "requires_bfs": False,
        },
        {
            "name": "PCA",
            "algorithm": DimensionalityReductionAlgorithm.PCA,
            "n_components": 103,
            "training_set_limit": 999999,
            "requires_bfs": False,
        },
        {
            "name": "LDA",
            "algorithm": DimensionalityReductionAlgorithm.LDA,
            "n_components": 9,  # max = class_count - 1 = 10 - 1 = 9
            "training_set_limit": 999999,
            "requires_bfs": False,
        },
        {
            "name": "Isomap",
            "algorithm": DimensionalityReductionAlgorithm.ISOMAP,
            "n_components": 50,
            "training_set_limit": 500,  # Mniejszy zbiór dla Isomap (wolniejszy)
            "requires_bfs": False,
        },
        {
            "name": "t-SNE",
            "algorithm": DimensionalityReductionAlgorithm.TSNE,
            "n_components": 3,
            "training_set_limit": 500,  # Mniejszy zbiór dla t-SNE (bardzo wolny)
            "requires_bfs": False,
        },
    ]

    # Dodaj FLOOD_FILL jeśli BFS jest dostępny
    if BFS_AVAILABLE:
        reduction_configs.append({
            "name": "Flood Fill (tradycyjna metoda)",
            "algorithm": DimensionalityReductionAlgorithm.FLOOD_FILL,
            "n_components": 50,  # Wynik flood fill
            "num_segments": 8,
            "pixel_normalization_rate": 0.2285805064971576,  # Optymalna wartość
            "flood_config": FloodConfig.from_string("1111"),
            "training_set_limit": 99999999,
            "requires_bfs": True,
        })
    else:
        print("⚠ Pomijanie FLOOD_FILL - wymaga BFS/numba")

    # ========================================================================
    # 4. URUCHOMIENIE TESTÓW
    # ========================================================================

    results = []

    for config in reduction_configs:
        print(f"\n{'='*60}")
        print(f"TESTOWANIE: {config['name']}")
        print(f"{'='*60}")

        # Tworzenie konfiguracji testu
        test_config = XGBTestConfig(
            # Parametry XGBoost
            **base_xgb_params,

            # Parametry datasetu
            class_count=dataset.class_count,  # Użyj class_count z datasetu
            image_size=28,

            # Parametry redukcji wymiarów
            dimensionality_reduction_algorithm=config["algorithm"],
            dimensionality_reduction_n_components=config["n_components"],
            training_set_limit=config["training_set_limit"],

            # Parametry wektorów
            num_segments=config.get("num_segments", 8),
            flood_config=config.get("flood_config", FloodConfig.from_string("1111")),
            pixel_normalization_rate=config.get("pixel_normalization_rate", 0.5),
        )

        try:
            # Uruchomienie testu
            runner = XGBTestRunner(
                train_dataset_path=dataset.train_path,
                test_dataset_path=dataset.test_path,
                train_data_type=dataset.data_type,
                test_data_type=dataset.data_type,
                train_labels_path=dataset.train_labels_path,
                test_labels_path=dataset.test_labels_path,
            )
            result = runner.run_tests([test_config])[0]  # Uruchom jeden test i weź pierwszy wynik

            # Zapisanie wyniku
            result_dict = {
                "name": config["name"],
                "algorithm": config["algorithm"].value if hasattr(config["algorithm"], 'value') else str(config["algorithm"]),
                "n_components": config["n_components"],
                "training_set_limit": config["training_set_limit"],
                "accuracy": result.accuracy,
                "training_time": result.training_time,
                "prediction_time": result.execution_time,
                "status": "success",
            }
            results.append(result_dict)

            print(f"✓ Wynik: Accuracy = {result.accuracy:.4f}")
            print(f"  Czas treningu: {result.training_time:.2f}s")
            print(f"  Czas predykcji: {result.execution_time:.2f}s")

        except Exception as e:
            print(f"✗ Błąd podczas testowania {config['name']}: {str(e)}")
            result_dict = {
                "name": config["name"],
                "algorithm": config["algorithm"].value if hasattr(config["algorithm"], 'value') else str(config["algorithm"]),
                "n_components": config["n_components"],
                "training_set_limit": config["training_set_limit"],
                "accuracy": 0.0,
                "training_time": 0.0,
                "prediction_time": 0.0,
                "status": f"error: {str(e)}",
            }
            results.append(result_dict)

    # ========================================================================
    # 5. PODSUMOWANIE WYNIKÓW
    # ========================================================================

    print(f"\n{'='*80}")
    print("PODSUMOWANIE WYNIKÓW PORÓWNANIA")
    print(f"{'='*80}")

    # Filtruj tylko udane wyniki
    successful_results = [r for r in results if r["status"] == "success"]

    if successful_results:
        # Sortowanie po dokładności (malejąco)
        results_sorted = sorted(successful_results, key=lambda x: x["accuracy"], reverse=True)

        print(f"{'Lp.':<2} {'Nazwa':<20} {'Accuracy':<10} {'Czas treningu':<15} {'Czas predykcji':<15}")
        print("-" * 80)

        for i, result in enumerate(results_sorted, 1):
            print(f"{i:<2} {result['name']:<20} {result['accuracy']:.4f} {result['training_time']:.2f}s {result['prediction_time']:.2f}s")

        print(f"\n{'='*80}")

        # Najlepszy wynik
        best_result = results_sorted[0]
        print(f"🏆 NAJLEPSZY WYNIK: {best_result['name']}")
        print(f"   Accuracy: {best_result['accuracy']:.4f}")
        print(f"   Czas treningu: {best_result['training_time']:.2f}s")
    else:
        print("❌ Brak udanych testów!")

    # Pokaż również nieudane testy
    failed_results = [r for r in results if r["status"] != "success"]
    if failed_results:
        print(f"\n❌ NIEUDANE TESTY:")
        for result in failed_results:
            print(f"   - {result['name']}: {result['status']}")


if __name__ == "__main__":
    main()