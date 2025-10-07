"""
SZABLON 2: Konfiguracja testów z zadanymi parametrami

Ten szablon pozwala na:
- Test konkretnych parametrów (bez optymalizacji)
- Wybór datasetów
- Możliwość testowania wielu zestawów parametrów naraz
"""

from typing import List, Dict
from Testers.Shared.configs import FloodConfig, TestRunnerConfig
from Testers.XgBoostTester.XGBTestRunner import XGBTestRunner
from Testers.XgBoostTester.configs import XGBTestConfig
from Testers.AnnTester.ANNTestRunner import ANNTestRunner
from Testers.AnnTester.configs import ANNTestConfig
from Testers.Shared.DataLoader import DataType
from Testers.BayesianOptimizer import (
    MNIST_DATASET,
    EMNIST_BALANCED_DATASET,
    EMNIST_DIGITS_DATASET,
    ARABIC_DATASET,
    USPS_DATASET,
    MNIST_C_BRIGHTNESS,
    MNIST_C_FOG,
    BASIC_DATASETS,
    DatasetConfig,
)


def main():
    """
    ========================================================================
    KONFIGURACJA TESTÓW Z ZADANYMI PARAMETRAMI
    ========================================================================
    """
    
    # ========================================================================
    # 1. WYBÓR ALGORYTMU
    # ========================================================================
    
    algorithm = "XGBoost"  # Opcje: "XGBoost", "ANN"
    
    
    # ========================================================================
    # 2. WYBÓR DATASETÓW
    # ========================================================================
    
    # OPCJA A: Ręczny wybór
    datasets = [
        MNIST_DATASET,
        EMNIST_DIGITS_DATASET,
        # ARABIC_DATASET,
        # USPS_DATASET,
    ]
    
    # OPCJA B: Użyj kolekcji
    # datasets = BASIC_DATASETS
    
    
    # ========================================================================
    # 3. DEFINICJA PARAMETRÓW DO TESTOWANIA
    # ========================================================================
    
    # Możesz zdefiniować jeden lub wiele zestawów parametrów
    
    if algorithm == "XGBoost":
        # Parametry dla XGBoost
        param_sets = [
            {
                "name": "Konfiguracja 1 - Optymalna z poprzedniej optymalizacji",
                "params": {
                    # Parametry wektoryzacji
                    "num_segments": 7,
                    "pixel_normalization_rate": 0.34,
                    "training_set_limit": 60000,
                    "flood_config": "1111",  # left, right, top, bottom
                    
                    # Parametry XGBoost
                    "n_estimators": 100,
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                }
            },
            # Możesz dodać więcej konfiguracji:
            # {
            #     "name": "Konfiguracja 2 - Eksperymentalna",
            #     "params": {
            #         "num_segments": 5,
            #         "pixel_normalization_rate": 0.25,
            #         "training_set_limit": 50000,
            #         "flood_config": "1110",
            #         "n_estimators": 150,
            #         "max_depth": 8,
            #         "learning_rate": 0.05,
            #         "subsample": 0.9,
            #         "colsample_bytree": 0.9,
            #     }
            # },
        ]
    
    else:  # ANN
        # Parametry dla ANN
        param_sets = [
            {
                "name": "Konfiguracja ANN 1",
                "params": {
                    # Parametry ANN
                    "trees_count": 2,
                    "leaves_count": 328,
                    
                    # Parametry wektoryzacji
                    "num_segments": 7,
                    "pixel_normalization_rate": 0.34,
                    "training_set_limit": 60000,
                    "flood_config": "1111",
                }
            },
        ]
    
    
    # ========================================================================
    # 4. OPCJE TESTOWANIA
    # ========================================================================
    
    # Czy zapisywać wyniki po każdym teście?
    save_after_each = True
    
    # Czy pominąć pierwszą generację wektorów? (jeśli już są)
    skip_first_vectors = False
    
    
    # ========================================================================
    # 5. WALIDACJA
    # ========================================================================
    
    if not datasets:
        print("BŁĄD: Nie wybrano żadnych datasetów!")
        return
    
    if not param_sets:
        print("BŁĄD: Nie zdefiniowano żadnych zestawów parametrów!")
        return
    
    
    # ========================================================================
    # 6. PODSUMOWANIE KONFIGURACJI
    # ========================================================================
    
    print("=" * 80)
    print("KONFIGURACJA TESTÓW Z ZADANYMI PARAMETRAMI")
    print("=" * 80)
    
    print(f"\n🤖 Algorytm: {algorithm}")
    
    print(f"\n📊 Datasety ({len(datasets)}):")
    for i, ds in enumerate(datasets, 1):
        print(f"  {i:2d}. {ds.display_name:40s} ({ds.class_count} classes, {ds.image_size}x{ds.image_size})")
    
    print(f"\n⚙️  Zestawy parametrów do przetestowania ({len(param_sets)}):")
    for i, param_set in enumerate(param_sets, 1):
        print(f"  {i}. {param_set['name']}")
    
    total_tests = len(datasets) * len(param_sets)
    print(f"\n📈 Szacunki:")
    print(f"  Liczba testów: {total_tests} (datasety × zestawy parametrów)")
    print(f"  Estimated time: {total_tests * 2:.0f} - {total_tests * 5:.0f} minut")
    
    print("=" * 80)
    
    
    # ========================================================================
    # 7. POTWIERDZENIE
    # ========================================================================
    
    if total_tests > 10:
        response = input("\n⚠️  Duża liczba testów. Kontynuować? (y/n): ")
        if response.lower() != 'y':
            print("❌ Testy anulowane")
            return
    
    
    # ========================================================================
    # 8. URUCHOMIENIE TESTÓW
    # ========================================================================
    
    print("\n🚀 Uruchamianie testów...")
    print("=" * 80)
    
    all_results = {}
    
    # Test runner config
    test_runner_config = TestRunnerConfig(
        skip_first_vector_generation=skip_first_vectors,
        save_results_after_each_test=save_after_each
    )
    
    # Dla każdego datasetu
    for dataset_idx, dataset in enumerate(datasets, 1):
        print(f"\n{'='*80}")
        print(f"DATASET {dataset_idx}/{len(datasets)}: {dataset.display_name}")
        print(f"{'='*80}")
        
        dataset_results = []
        
        # Utwórz test runner dla tego datasetu
        if algorithm == "XGBoost":
            runner = XGBTestRunner(
                train_dataset_path=dataset.train_path,
                test_dataset_path=dataset.test_path,
                train_data_type=dataset.data_type,
                test_data_type=dataset.data_type,
                train_labels_path=dataset.train_labels_path,
                test_labels_path=dataset.test_labels_path,
                config=test_runner_config
            )
        else:  # ANN
            runner = ANNTestRunner(
                train_dataset_path=dataset.train_path,
                test_dataset_path=dataset.test_path,
                train_data_type=dataset.data_type,
                test_data_type=dataset.data_type,
                train_labels_path=dataset.train_labels_path,
                test_labels_path=dataset.test_labels_path,
                config=test_runner_config
            )
        
        # Test każdego zestawu parametrów
        test_configs = []
        for param_set in param_sets:
            params = param_set['params']
            
            # Utwórz test config
            if algorithm == "XGBoost":
                test_config = XGBTestConfig(
                    # Wektoryzacja
                    num_segments=params['num_segments'],
                    pixel_normalization_rate=params['pixel_normalization_rate'],
                    training_set_limit=params['training_set_limit'],
                    flood_config_str=params['flood_config'],
                    
                    # Dataset info
                    class_count=dataset.class_count,
                    image_size=dataset.image_size,
                    
                    # XGBoost specific
                    n_estimators=params['n_estimators'],
                    max_depth=params['max_depth'],
                    learning_rate=params['learning_rate'],
                    subsample=params['subsample'],
                    colsample_bytree=params['colsample_bytree'],
                )
            else:  # ANN
                flood_config = FloodConfig.from_string(params['flood_config'])
                test_config = ANNTestConfig(
                    # ANN
                    trees_count=params['trees_count'],
                    leaves_count=params['leaves_count'],
                    
                    # Wektoryzacja
                    num_segments=params['num_segments'],
                    pixel_normalization_rate=params['pixel_normalization_rate'],
                    training_set_limit=params['training_set_limit'],
                    flood_config=flood_config,
                    
                    # Dataset info
                    class_count=dataset.class_count,
                    image_size=dataset.image_size,
                )
            
            test_configs.append(test_config)
        
        # Uruchom testy
        results = runner.run_tests(test_configs)
        
        # Zapisz wyniki
        for param_set, result in zip(param_sets, results):
            dataset_results.append({
                'param_set_name': param_set['name'],
                'params': param_set['params'],
                'result': result
            })
        
        all_results[dataset.display_name] = dataset_results
    
    
    # ========================================================================
    # 9. PODSUMOWANIE WYNIKÓW
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("✅ WSZYSTKIE TESTY ZAKOŃCZONE")
    print("=" * 80)
    
    print("\n📊 PODSUMOWANIE WYNIKÓW:\n")
    
    for dataset_name, dataset_results in all_results.items():
        print(f"\n{dataset_name}:")
        print("-" * 80)
        
        for result_data in dataset_results:
            result = result_data['result']
            print(f"\n  {result_data['param_set_name']}:")
            print(f"    Accuracy:     {result.accuracy:.4f}")
            print(f"    Precision:    {result.precision:.4f}")
            print(f"    Recall:       {result.recall:.4f}")
            print(f"    F1 Score:     {result.f1_score:.4f}")
            print(f"    Total time:   {result.total_execution_time:.2f}s")
        
        # Znajdź najlepszy
        best = max(dataset_results, key=lambda x: x['result'].accuracy)
        print(f"\n  ⭐ NAJLEPSZY: {best['param_set_name']} (Accuracy: {best['result'].accuracy:.4f})")
    
    print("\n" + "=" * 80)
    print("📁 Wyniki zapisane w: results/")
    print("=" * 80)


if __name__ == "__main__":
    """
    ========================================================================
    QUICK START GUIDE:
    ========================================================================
    
    1. Wybierz algorytm (sekcja 1): "XGBoost" lub "ANN"
    2. Wybierz datasety (sekcja 2)
    3. Zdefiniuj parametry do testowania (sekcja 3)
    4. Uruchom: python template_test_params.py
    5. Sprawdź wyniki w folderze: results/
    
    ========================================================================
    PRZYKŁADOWE UŻYCIA:
    ========================================================================
    
    TEST POJEDYNCZEJ KONFIGURACJI:
    - algorithm = "XGBoost"
    - datasets = [MNIST_DATASET]
    - param_sets = [jeden zestaw parametrów]
    
    PORÓWNANIE KONFIGURACJI:
    - datasets = [MNIST_DATASET]
    - param_sets = [config1, config2, config3]
    → Porówna 3 konfiguracje na MNIST
    
    TEST NA WIELU DATASETACH:
    - datasets = BASIC_DATASETS
    - param_sets = [najlepsza konfiguracja]
    → Test tej samej konfiguracji na 5 datasetach
    
    COMPREHENSIVE COMPARISON:
    - datasets = [MNIST, EMNIST, USPS]
    - param_sets = [optimal_config, experimental_config]
    → 3 datasety × 2 konfiguracje = 6 testów
    
    ========================================================================
    """
    main()
