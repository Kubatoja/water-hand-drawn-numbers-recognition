import time
import numpy as np
from typing import List, Optional

from Testers.Shared.DataLoader import DataLoader, DataType
from Testers.Shared.models import TestResult, VectorNumberData
from Testers.Shared.TestResultCollector import TestResultCollector
from Testers.Shared.VectorManager import VectorManager
from Testers.Shared.configs import TestRunnerConfig, DimensionalityReductionAlgorithm
from Testers.Shared.base_test_runner import BaseTestRunner
from Testers.SVMTester.configs import SVMTestConfig
from Testers.SVMTester.SVMTester import SVMTester


class SVMTestRunner(BaseTestRunner):
    """Test runner dla SVM używający wspólnych komponentów"""

    def __init__(
        self,
        train_dataset_path: str,
        test_dataset_path: str,
        train_data_type: DataType = DataType.MNIST_FORMAT,
        test_data_type: DataType = DataType.MNIST_FORMAT,
        train_labels_path: Optional[str] = None,
        test_labels_path: Optional[str] = None,
        config: TestRunnerConfig = TestRunnerConfig(),
        vectors_file: str = "Data/vectors.csv",
        external_collector: Optional[TestResultCollector] = None
    ):
        super().__init__(
            train_dataset_path=train_dataset_path,
            test_dataset_path=test_dataset_path,
            train_data_type=train_data_type,
            test_data_type=test_data_type,
            train_labels_path=train_labels_path,
            test_labels_path=test_labels_path,
            config=config,
            vectors_file=vectors_file,
            external_collector=external_collector
        )

    def get_algorithm_name(self) -> str:
        return "SVM"

    def get_tester_instance(self, num_classes: int):
        return SVMTester(num_classes=num_classes)