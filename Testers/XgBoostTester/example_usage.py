"""
Przykładowy skrypt demonstracyjny dla XGBoost Testera
"""

from Testers.XgBoostTester.XGBTestRunner import XGBTestRunner
from Testers.XgBoostTester.configs import XGBTestConfig, XGBTestConfigField, FieldConfig
from Testers.XgBoostTester.TestConfigFactory import create_xgb_test_configs
from Testers.Shared.configs import FloodConfig, TestRunnerConfig
from Testers.Shared.DataLoader import DataType


def example_single_test():
    """Przykład pojedynczego testu XGBoost"""
    print("=" * 60)
    print("PRZYKŁAD 1: Pojedynczy test XGBoost")
    print("=" * 60)
    
    config = XGBTestConfig(
        # Parametry XGBoost
        learning_rate=0.1,
        n_estimators=100,
        max_depth=6,
        min_child_weight=1.0,
        gamma=0.0,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        
        # Parametry wektorów
        num_segments=4,
        pixel_normalization_rate=0.5,
        training_set_limit=1000,
        flood_config=FloodConfig.from_string("1111"),
        
        # Dataset info
        class_count=10
    )
    
    runner = XGBTestRunner(
        train_dataset_path="Data/mnist_train.csv",
        test_dataset_path="Data/mnist_test.csv",
        train_data_type=DataType.MNIST_FORMAT,
        test_data_type=DataType.MNIST_FORMAT,
        config=TestRunnerConfig(
            skip_first_vector_generation=False,
            save_results_after_each_test=True
        )
    )
    
    results = runner.run_tests([config])
    
    if results:
        result = results[0]
        print(f"\nWyniki:")
        print(f"  Accuracy:  {result.accuracy:.4f}")
        print(f"  Precision: {result.precision:.4f}")
        print(f"  Recall:    {result.recall:.4f}")
        print(f"  F1-Score:  {result.f1_score:.4f}")


def example_parameter_sweep():
    """Przykład przeszukiwania przestrzeni parametrów"""
    print("\n" + "=" * 60)
    print("PRZYKŁAD 2: Przeszukiwanie parametrów learning_rate")
    print("=" * 60)
    
    default_config = XGBTestConfig(
        learning_rate=0.1,
        n_estimators=50,
        max_depth=5,
        min_child_weight=1.0,
        gamma=0.0,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        num_segments=4,
        pixel_normalization_rate=0.5,
        training_set_limit=500,  # Mały zbiór dla szybkiego testu
        flood_config=FloodConfig.from_string("1111"),
        class_count=10
    )
    
    field_configs = [
        FieldConfig(
            field_name=XGBTestConfigField.LEARNING_RATE,
            start=0.05,
            stop=0.3,
            step=0.05
        )
    ]
    
    test_configs = create_xgb_test_configs(
        field_configs=field_configs,
        generate_combinations=False,
        default_config=default_config
    )
    
    print(f"Wygenerowano {len(test_configs)} konfiguracji testowych")
    
    runner = XGBTestRunner(
        train_dataset_path="Data/mnist_train.csv",
        test_dataset_path="Data/mnist_test.csv"
    )
    
    results = runner.run_tests(test_configs)
    
    # Znajdź najlepszy wynik
    if results:
        best_result = max(results, key=lambda r: r.accuracy)
        print(f"\nNajlepszy wynik:")
        print(f"  Learning rate: {best_result.config.learning_rate}")
        print(f"  Accuracy:      {best_result.accuracy:.4f}")
        print(f"  F1-Score:      {best_result.f1_score:.4f}")


def example_multi_parameter_combinations():
    """Przykład testowania kombinacji parametrów"""
    print("\n" + "=" * 60)
    print("PRZYKŁAD 3: Kombinacje max_depth i n_estimators")
    print("=" * 60)
    
    default_config = XGBTestConfig(
        learning_rate=0.1,
        n_estimators=50,
        max_depth=5,
        min_child_weight=1.0,
        gamma=0.0,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        num_segments=4,
        pixel_normalization_rate=0.5,
        training_set_limit=500,
        flood_config=FloodConfig.from_string("1111"),
        class_count=10
    )
    
    field_configs = [
        FieldConfig(
            field_name=XGBTestConfigField.MAX_DEPTH,
            start=3,
            stop=7,
            step=2
        ),
        FieldConfig(
            field_name=XGBTestConfigField.N_ESTIMATORS,
            start=50,
            stop=100,
            step=50
        )
    ]
    
    # generate_combinations=True dla wszystkich kombinacji
    test_configs = create_xgb_test_configs(
        field_configs=field_configs,
        generate_combinations=True,
        default_config=default_config
    )
    
    print(f"Wygenerowano {len(test_configs)} konfiguracji testowych (kombinacje)")
    
    runner = XGBTestRunner(
        train_dataset_path="Data/mnist_train.csv",
        test_dataset_path="Data/mnist_test.csv",
        config=TestRunnerConfig(save_results_after_each_test=True)
    )
    
    results = runner.run_tests(test_configs)
    
    print(f"\nPrzetestowano {len(results)} konfiguracji")


if __name__ == "__main__":
    # Uruchom przykłady
    # UWAGA: Odkomentuj odpowiedni przykład
    
    # Przykład 1: Pojedynczy test
    # example_single_test()
    
    # Przykład 2: Przeszukiwanie jednego parametru
    # example_parameter_sweep()
    
    # Przykład 3: Kombinacje parametrów
    # example_multi_parameter_combinations()
    
    print("\n" + "=" * 60)
    print("Odkomentuj wybrany przykład w __main__ aby uruchomić")
    print("=" * 60)
