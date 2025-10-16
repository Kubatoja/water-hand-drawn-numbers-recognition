from typing import Optional, TYPE_CHECKING

from Testers.Shared.DataLoader import DataType
from Testers.Shared.configs import TestRunnerConfig
from Testers.KNNTester.KNNTestRunner import KNNTestRunner
from Testers.XgBoostTester.XGBTestRunner import XGBTestRunner
from Testers.MLPTester.MLPTestRunner import MLPTestRunner
from Testers.SVMTester.SVMTestRunner import SVMTestRunner

if TYPE_CHECKING:
    from Testers.Shared.base_test_runner import BaseTestRunner


class ClassifierType:
    """Typy klasyfikatorów"""
    KNN = "knn"
    XGBOOST = "xgboost"
    MLP = "mlp"
    SVM = "svm"


class TestRunnerFactory:
    """Fabryka do tworzenia TestRunner'ów na podstawie typu klasyfikatora"""

    @staticmethod
    def create_runner(
        classifier_type: str,
        train_dataset_path: str,
        test_dataset_path: str,
        train_data_type: DataType = DataType.MNIST_FORMAT,
        test_data_type: DataType = DataType.MNIST_FORMAT,
        train_labels_path: Optional[str] = None,
        test_labels_path: Optional[str] = None,
        config: TestRunnerConfig = TestRunnerConfig(),
        vectors_file: str = "Data/vectors.csv"
    ) -> 'BaseTestRunner':
        """
        Tworzy odpowiedni TestRunner na podstawie typu klasyfikatora

        Args:
            classifier_type: Typ klasyfikatora (np. 'knn', 'xgboost')
            ... inne parametry jak w BaseTestRunner

        Returns:
            Instancja odpowiedniego TestRunner'a

        Raises:
            ValueError: Jeśli classifier_type nie jest obsługiwany
        """
        if classifier_type == ClassifierType.KNN:
            return KNNTestRunner(
                train_dataset_path=train_dataset_path,
                test_dataset_path=test_dataset_path,
                train_data_type=train_data_type,
                test_data_type=test_data_type,
                train_labels_path=train_labels_path,
                test_labels_path=test_labels_path,
                config=config,
                vectors_file=vectors_file
            )
        elif classifier_type == ClassifierType.XGBOOST:
            return XGBTestRunner(
                train_dataset_path=train_dataset_path,
                test_dataset_path=test_dataset_path,
                train_data_type=train_data_type,
                test_data_type=test_data_type,
                train_labels_path=train_labels_path,
                test_labels_path=test_labels_path,
                config=config,
                vectors_file=vectors_file
            )
        elif classifier_type == ClassifierType.MLP:
            return MLPTestRunner(
                train_dataset_path=train_dataset_path,
                test_dataset_path=test_dataset_path,
                train_data_type=train_data_type,
                test_data_type=test_data_type,
                train_labels_path=train_labels_path,
                test_labels_path=test_labels_path,
                config=config,
                vectors_file=vectors_file
            )
        elif classifier_type == ClassifierType.SVM:
            return SVMTestRunner(
                train_dataset_path=train_dataset_path,
                test_dataset_path=test_dataset_path,
                train_data_type=train_data_type,
                test_data_type=test_data_type,
                train_labels_path=train_labels_path,
                test_labels_path=test_labels_path,
                config=config,
                vectors_file=vectors_file
            )
        else:
            raise ValueError(f"Nieobsługiwany typ klasyfikatora: {classifier_type}")