import time
from typing import List

from Tester.ANNTester import ANNTestConfig, ANNTester
from Tester.DataLoader import DataType, DataLoader
from Tester.otherModels import TestResult
from Tester.TestResultCollector import TestResultCollector
from Tester.VectorManager import VectorManager
from Tester.configs import TestRunnerConfig
from VectorSearch.annoy import Ann

class ANNTestRunner:
    """Updated test runner using the merged VectorManager"""

    def __init__(self, train_dataset_path: str, test_dataset_path: str,
                 train_data_type: DataType = DataType.MNIST_FORMAT,
                 test_data_type: DataType = DataType.MNIST_FORMAT,
                 config: TestRunnerConfig = TestRunnerConfig(),
                 vectors_file: str = "Data/vectors.csv"):

        self.config = config
        self.loader = DataLoader()
        self.result_collector = TestResultCollector()

        # Use merged VectorManager
        self.vector_manager = VectorManager(vectors_file)

        # Load data using DataLoader
        self.train_data = self.loader.load_data(train_dataset_path, train_data_type)
        self.test_data = self.loader.load_data(test_dataset_path, test_data_type)

    def run_tests(self, test_configs: List['ANNTestConfig']) -> List['TestResult']:
        """Run all tests using the merged VectorManager"""

        for index, test_config in enumerate(test_configs):
            print(f"Starting test case #{index + 1}/{len(test_configs)}")

            try:
                result = self._run_single_test(test_config, index)

                # Save results incrementally or just add to collection
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

        # Print summary and save final results
        self.result_collector.print_summary(len(test_configs))

        if not self.config.save_results_after_each_test:
            # Save all results at once if not saving incrementally
            self.result_collector.save_results()

        # Create final summary report
        try:
            report_path = self.result_collector.create_final_report()
            print(f"Final summary report created: {report_path}")
        except ValueError:
            print("No results to create final report")

        return self.result_collector.results

    def _run_single_test(self, test_config: ANNTestConfig, test_index: int) -> TestResult:
        """Run a single test"""
        # Get training vectors using merged VectorManager
        is_first_test = (test_index == 0)
        force_regenerate = not (self.config.skip_first_vector_generation and is_first_test)

        training_vectors = self.vector_manager.get_training_vectors(
            self.train_data,
            test_config,
            force_regenerate=force_regenerate,
            auto_save=True
        )

        # Validate vectors
        if not self.vector_manager.validate_vectors(training_vectors):
            raise ValueError("Vector validation failed")

        # Rest of the test logic...
        start_time = time.perf_counter()

        print("Building ANN forest...")
        ann_forest = Ann(
            training_vectors,
            test_config.trees_count,
            test_config.leaves_count,
            0.95
        )

        training_time = time.perf_counter() - start_time

        print("Running test evaluation...")
        tester = ANNTester(num_classes=test_config.class_count)
        result = tester.test_model(ann_forest, self.test_data, test_config)

        # Fill result
        result.train_set_size = len(training_vectors)
        result.test_set_size = len(self.test_data)
        result.training_time = training_time
        result.config = test_config

        return result