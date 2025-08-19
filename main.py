from Tester.ANNTestRunner import ANNTestRunner
from Tester.DataLoader import DataType
from Tester.TestConfigFactory import create_ann_test_configs
from Tester.configs import TestRunnerConfig, ANNTestConfig, FloodConfig, ANNTestConfigField, FieldConfig

if __name__ == "__main__":
    # Inicjalizacja test runnera
    test_runner_config = TestRunnerConfig(skip_first_vector_generation=False, save_results_after_each_test=True)

    # train_dataset = 'Datasets/FashionMnist/fashion-mnist_train.csv'
    train_dataset = 'Data/usps_train.csv'
    # train_dataset = 'Data/mnist_train.csv'
    # train_dataset = 'Datasets/USPS/usps_test.csv'
    # train_dataset = 'Datasets/KMNIST/kmnist_train.csv'
    train_datatype = DataType.MNIST_FORMAT
    # test_dataset = 'Datasets/FashionMnist/fashion-mnist_test.csv'
    test_dataset = 'Data/usps_test.csv'
    # test_dataset = 'Data/mnist_test.csv'
    # test_dataset = 'Datasets/USPS/usps_test.csv'
    # test_dataset = 'Datasets/KMNIST/kmnist_test.csv'
    test_datatype = DataType.MNIST_FORMAT


    ann_test_runner = ANNTestRunner(
        train_dataset_path=train_dataset,
        test_dataset_path=test_dataset,
        train_data_type=train_datatype,
        test_data_type=test_datatype,
        config=test_runner_config
    )

    default_confg = ANNTestConfig(
        trees_count=2,
        leaves_count=328,
        training_set_limit=8000,  # Full test ultra-optimizations
        num_segments=7,
        pixel_normalization_rate=0.34,
        flood_config= FloodConfig(True, True, True, True),

        class_count=10, ## to jak będziemy inne datasety które nie mają 10 cyfr to tu zmioenić
        enable_centering=False  # Wyłączamy centrowanie - obniżało wyniki
    )

    # to ci wygeneruje wszystkie kombinacje tych parametrów, jeżeli nie chcesz kombinacji ustaw generate_combinations = False
    configs_to_generate = [
    # train_size
]

    generate_combinations = False

    # Generujemy wszystkie możliwe kombinacje
        # Nadpisujemy generated_configs na listę testów dla wszystkich wartości z zapytania
    generated_configs = []

        # 1. training_set_limit
    for train_size in [100, 500, 1000, 2500, 5000, 8000]:
        generated_configs.append(ANNTestConfig(
            trees_count=2,
            leaves_count=328,
            training_set_limit=train_size,
            num_segments=7,
            pixel_normalization_rate=0.34,
            flood_config=FloodConfig(True, True, True, True),
            class_count=10,
            enable_centering=False
        ))

        # 2. pixel_normalization_rate
    for norm_rate in [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
        generated_configs.append(ANNTestConfig(
            trees_count=2,
            leaves_count=328,
            training_set_limit=8000,
            num_segments=7,
            pixel_normalization_rate=norm_rate,
            flood_config=FloodConfig(True, True, True, True),
            class_count=10,
            enable_centering=False
        ))
    # 3. trees_count
    for trees in range(1, 11):
        generated_configs.append(ANNTestConfig(
            trees_count=trees,
            leaves_count=328,
            training_set_limit=8000,
            num_segments=7,
            pixel_normalization_rate=0.34,
            flood_config=FloodConfig(True, True, True, True),
            class_count=10,
            enable_centering=False
        ))

        ann_test_runner.run_tests(generated_configs)





