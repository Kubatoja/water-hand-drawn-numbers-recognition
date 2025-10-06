from Tester.ANNTestRunner import ANNTestRunner
from Tester.DataLoader import DataType
from Tester.TestConfigFactory import create_ann_test_configs
from Tester.configs import TestRunnerConfig, ANNTestConfig, FloodConfig, ANNTestConfigField, FieldConfig

if __name__ == "__main__":
    # Inicjalizacja test runnera
    test_runner_config = TestRunnerConfig(skip_first_vector_generation=False, save_results_after_each_test=True)

    # train_dataset = 'Datasets/FashionMnist/fashion-mnist_train.csv'
    train_dataset = 'Data/mnist_train.csv'
    # train_dataset = 'Datasets/USPS/usps_train.csv'
    # train_dataset = 'Datasets/KMNIST/kmnist_train.csv'
    train_datatype = DataType.MNIST_FORMAT
    # test_dataset = 'Datasets/FashionMnist/fashion-mnist_test.csv'
    test_dataset = 'Data/mnist_test.csv'
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
        training_set_limit=60000,  # Full test ultra-optimizations
        num_segments=7,
        pixel_normalization_rate= 0.34,
        flood_config= FloodConfig(True, True, True, True),

        class_count=10 ## to jak będziemy inne datasety które nie mają 10 cyfr to tu zmioenić
    )

    # to ci wygeneruje wszystkie kombinacje tych parametrów, jeżeli nie chcesz kombinacji ustaw generate_combinations = False
    configs_to_generate = [
        FieldConfig(ANNTestConfigField.NUM_SEGMENTS, start=7, stop=7, step=2),
    ]

    generate_combinations = True

    # Generujemy wszystkie możliwe kombinacje
    generated_configs = create_ann_test_configs(configs_to_generate, generate_combinations, default_confg )

    ann_test_runner.run_tests(generated_configs)





