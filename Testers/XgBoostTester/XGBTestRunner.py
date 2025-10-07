import time
from typing import List, Optional

from Testers.Shared.DataLoader import DataLoader, DataType
from Testers.Shared.models import TestResult
from Testers.Shared.TestResultCollector import TestResultCollector
from Testers.Shared.VectorManager import VectorManager
from Testers.Shared.configs import TestRunnerConfig
from Testers.XgBoostTester.configs import XGBTestConfig
from Testers.XgBoostTester.XGBTester import XGBTester


class XGBTestRunner:
    """Test runner dla XGBoost używający wspólnych komponentów"""

    def __init__(
        self,
        train_dataset_path: str,
        test_dataset_path: str,
        train_data_type: DataType = DataType.MNIST_FORMAT,
        test_data_type: DataType = DataType.MNIST_FORMAT,
        train_labels_path: Optional[str] = None,  # Dla SEPARATED_FORMAT
        test_labels_path: Optional[str] = None,   # Dla SEPARATED_FORMAT
        config: TestRunnerConfig = TestRunnerConfig(),
        vectors_file: str = "Data/vectors.csv"
    ):
        self.config = config
        self.loader = DataLoader()
        self.result_collector = TestResultCollector(algorithm_name="XGBoost")

        # Vector manager
        self.vector_manager = VectorManager(vectors_file)

        # Wczytaj dane
        self.train_data = self.loader.load_data(
            train_dataset_path, 
            train_data_type,
            labels_path=train_labels_path
        )
        self.test_data = self.loader.load_data(
            test_dataset_path, 
            test_data_type,
            labels_path=test_labels_path
        )

    def run_tests(self, test_configs: List[XGBTestConfig]) -> List[TestResult]:
        """Uruchamia wszystkie testy używając wspólnego VectorManager"""

        for index, test_config in enumerate(test_configs):
            print(f"Starting test case #{index + 1}/{len(test_configs)}")

            try:
                result = self._run_single_test(test_config, index)

                if self.config.save_results_after_each_test:
                    self.result_collector.add_success_and_save(result, index)
                    print(f"Test case #{index + 1} completed and saved")
                else:
                    self.result_collector.add_success(result)
                    print(f"Test case #{index + 1} completed successfully")

            except Exception as e:
                error_msg = str(e)
                print(f"Test case #{index + 1} failed: {error_msg}")
                self.result_collector.add_failure(index, error_msg)

            print("-" * 40)

        # Podsumowanie i zapis wyników
        self.result_collector.print_summary(len(test_configs))

        if not self.config.save_results_after_each_test:
            self.result_collector.save_results()

        # Raport końcowy
        try:
            report_path = self.result_collector.create_final_report()
            print(f"Final summary report created: {report_path}")
        except ValueError:
            print("No results to create final report")

        return self.result_collector.results

    def _run_single_test(self, test_config: XGBTestConfig, test_index: int) -> TestResult:
        """Uruchamia pojedynczy test"""
        total_execution_start = time.perf_counter()
        
        # Generowanie wektorów
        vector_generation_start = time.perf_counter()
        
        is_first_test = (test_index == 0)
        force_regenerate = not (self.config.skip_first_vector_generation and is_first_test)

        training_vectors = self.vector_manager.get_training_vectors(
            self.train_data,
            test_config,
            force_regenerate=force_regenerate,
            auto_save=True
        )

        # Walidacja wektorów
        if not self.vector_manager.validate_vectors(training_vectors):
            raise ValueError("Vector validation failed")

        # Przygotuj wektory testowe
        print("Preparing test vectors...")
        test_vectors = []
        for test_number in self.test_data:
            test_number.binarize_data(test_config.pixel_normalization_rate)
            vector = VectorManager.create_vector_for_single_sample(test_number, test_config)
            test_vectors.append(vector)

        vector_generation_time = time.perf_counter() - vector_generation_start
        
        print(f"📊 Vector generation time: {vector_generation_time:.3f}s")

        # Trenowanie i testowanie
        print("Training and testing XGBoost model...")
        tester = XGBTester(num_classes=test_config.class_count)
        
        model, result = tester.train_and_test(
            training_vectors,
            test_vectors,
            test_config
        )

        # Uzupełnij wynik
        total_execution_time = time.perf_counter() - total_execution_start
        
        print(f"📊 Total execution time: {total_execution_time:.3f}s")
        print(f"📊 Testing time: {result.execution_time:.3f}s")

        result.train_set_size = len(training_vectors)
        result.test_set_size = len(test_vectors)
        result.execution_time = total_execution_time  # Całkowity czas
        result.config = test_config

        return result
