import time
from typing import List

import numpy as np

from Testers.Shared.models import TestResult, VectorNumberData
from Testers.Shared.MetricsCalculator import MetricsCalculator
from Testers.TabPFNTester.configs import TabPFNTestConfig


def _resolve_device(device: str) -> str:
    """Wykrywa dostępne urządzenie."""
    if device == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return device


class TabPFNTester:
    """Klasa odpowiedzialna za testowanie modeli TabPFN-v2"""

    def __init__(self, num_classes: int = 10):
        self.num_classes = num_classes
        self.metrics_calculator = MetricsCalculator()

    def test_model(
        self,
        model,
        test_vectors: List[VectorNumberData],
        config: TabPFNTestConfig
    ) -> TestResult:
        """
        Testuje model TabPFN na podanych danych testowych.

        Args:
            model: Wytrenowany model TabPFN
            test_vectors: Lista wektorów testowych
            config: Konfiguracja testu

        Returns:
            TestResult: Wyniki testowania
        """
        if not test_vectors:
            raise ValueError("Lista test_vectors nie może być pusta")

        start_time = time.perf_counter()

        X_test = np.array([vec.vector for vec in test_vectors])
        y_test = np.array([vec.label for vec in test_vectors])

        y_pred = model.predict(X_test)

        end_time = time.perf_counter()
        execution_time = end_time - start_time

        actual_labels = y_test.astype(int)
        predicted_labels = np.array(y_pred).astype(int)

        correct_predictions = np.sum(actual_labels == predicted_labels)
        total_predictions = len(test_vectors)
        incorrect_predictions = total_predictions - correct_predictions
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0

        metrics = self.metrics_calculator.calculate_all_metrics(
            actual_labels, predicted_labels, self.num_classes
        )
        confusion_matrix = self.metrics_calculator.calculate_confusion_matrix(
            actual_labels, predicted_labels, self.num_classes
        )

        results = TestResult(
            execution_time=execution_time,
            correct_predictions=int(correct_predictions),
            incorrect_predictions=int(incorrect_predictions),
            accuracy=accuracy,
            confusion_matrix=confusion_matrix,
            precision=metrics['precision'],
            recall=metrics['recall'],
            f1_score=metrics['f1_score'],
            per_class_precision=metrics['per_class_precision'],
            per_class_recall=metrics['per_class_recall'],
            per_class_f1=metrics['per_class_f1'],
            config=None,
            training_time=None,
            train_set_size=None,
            test_set_size=None
        )

        self._print_results(results)
        return results

    def train_and_test(
        self,
        training_vectors: List[VectorNumberData],
        test_vectors: List[VectorNumberData],
        config: TabPFNTestConfig
    ) -> tuple:
        """
        Trenuje i testuje model TabPFN-v2.

        Args:
            training_vectors: Lista wektorów treningowych
            test_vectors: Lista wektorów testowych
            config: Konfiguracja testu

        Returns:
            Tuple(model, wyniki): Wytrenowany model i wyniki testowania
        """
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion

        X_train = np.array([vec.vector for vec in training_vectors])
        y_train = np.array([vec.label for vec in training_vectors])

        device = _resolve_device(config.device)
        print(f"Training TabPFN-v2 model on device='{device}'...")

        train_start = time.perf_counter()

        if config.use_v2:
            model = TabPFNClassifier.create_default_for_version(
                ModelVersion.V2,
                n_estimators=config.n_estimators,
                device=device,
            )
        else:
            model = TabPFNClassifier(
                n_estimators=config.n_estimators,
                device=device,
            )

        model.fit(X_train, y_train)

        train_end = time.perf_counter()
        training_time = train_end - train_start
        print(f"📊 Training completed in {training_time:.3f}s")

        print("Running test evaluation...")
        result = self.test_model(model, test_vectors, config)
        result.training_time = training_time

        return model, result

    def _print_results(self, results: TestResult) -> None:
        """Wyświetla wyniki testowania"""
        print(f"Test zakończony:")
        print(f"  Poprawne predykcje: {results.correct_predictions}")
        print(f"  Niepoprawne predykcje: {results.incorrect_predictions}")
        print(f"  Dokładność: {results.accuracy:.2%}")
        print(f"  Precision: {results.precision:.4f}")
        print(f"  Recall: {results.recall:.4f}")
        print(f"  F1-Score: {results.f1_score:.4f}")
        print(f"  Czas wykonania: {results.execution_time:.3f}s")
