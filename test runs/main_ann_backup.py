"""
Przykładowy plik z konfiguracją testów ANN
(backup z oryginalnego main.py)
"""

from Testers.AnnTester.ANNTestRunner import ANNTestRunner
from Testers.Shared.DataLoader import DataType
from Testers.AnnTester.TestConfigFactory import create_ann_test_configs
from Testers.Shared.configs import TestRunnerConfig, FloodConfig
from Testers.AnnTester.configs import ANNTestConfig, ANNTestConfigField, FieldConfig

if __name__ == "__main__":
    # Inicjalizacja test runnera
    test_runner_config = TestRunnerConfig(
        skip_first_vector_generation=False, 
        save_results_after_each_test=True
    )

    # Konfiguracja datasetu
    train_dataset = 'Data/mnist_train.csv'
    train_datatype = DataType.MNIST_FORMAT
    test_dataset = 'Data/mnist_test.csv'
    test_datatype = DataType.MNIST_FORMAT

    ann_test_runner = ANNTestRunner(
        train_dataset_path=train_dataset,
        test_dataset_path=test_dataset,
        train_data_type=train_datatype,
        test_data_type=test_datatype,
        config=test_runner_config
    )

    default_config = ANNTestConfig(
        trees_count=2,
        leaves_count=328,
        training_set_limit=60000,  # Full test ultra-optimizations
        num_segments=7,
        pixel_normalization_rate=0.34,
        flood_config=FloodConfig(True, True, True, True),
        class_count=10
    )

    # Konfiguracje do wygenerowania
    configs_to_generate = [
        FieldConfig(ANNTestConfigField.NUM_SEGMENTS, start=7, stop=7, step=2),
    ]

    generate_combinations = True

    # Generujemy wszystkie możliwe kombinacje
    generated_configs = create_ann_test_configs(
        configs_to_generate, 
        generate_combinations, 
        default_config
    )

    print(f"🚀 Uruchamianie {len(generated_configs)} testów ANN...")
    ann_test_runner.run_tests(generated_configs)
