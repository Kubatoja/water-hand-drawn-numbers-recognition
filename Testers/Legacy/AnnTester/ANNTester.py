import time
from typing import List

import numpy as np
from Testers.Shared.VectorManager import VectorManager
from Testers.Shared.MetricsCalculator import MetricsCalculator
from Testers.Shared.models import TestResult, RawNumberData
from Testers.AnnTester.configs import ANNTestConfig
from VectorSearch.annoy import Ann


class ANNTester:
    """Klasa odpowiedzialna za testowanie modeli ANN."""

    def __init__(self, num_classes: int = 10):
        self.num_classes = num_classes
        self.metrics_calculator = MetricsCalculator()

    def test_model(
            self,
            ann_forest: Ann,
            test_numbers: List[RawNumberData],
            config: ANNTestConfig
    ) -> TestResult:
        """
        Testuje model ANN na podanych danych testowych.

        Args:
            ann_forest: Model ANN do testowania
            test_numbers: Lista danych testowych
            config: Konfiguracja testu

        Returns:
            TestResults: Wyniki testowania
        """
        if not test_numbers:
            raise ValueError("Lista test_numbers nie może być pusta")

        confusion_matrix_data = np.zeros((self.num_classes, self.num_classes), dtype=int)
        correct_predictions = 0

        start_time = time.perf_counter()

        actual_labels = []
        predicted_labels = []

        for test_number in test_numbers:
            prediction = self._predict_single_sample(test_number, ann_forest, config)

            actual_labels.append(int(test_number.label))
            predicted_labels.append(int(prediction))

            if prediction == test_number.label:
                correct_predictions += 1

            confusion_matrix_data[int(test_number.label)][int(prediction)] += 1

        end_time = time.perf_counter()
        execution_time = end_time - start_time

        total_predictions = len(test_numbers)
        incorrect_predictions = total_predictions - correct_predictions
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0

        # Oblicz rozszerzone metryki
        actual_labels_np = np.array(actual_labels)
        predicted_labels_np = np.array(predicted_labels)
        
        metrics = self.metrics_calculator.calculate_all_metrics(
            actual_labels_np, predicted_labels_np, self.num_classes
        )

        results = TestResult(
            execution_time=execution_time,
            correct_predictions=correct_predictions,
            incorrect_predictions=incorrect_predictions,
            accuracy=accuracy,
            confusion_matrix=confusion_matrix_data,
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

    def _predict_single_sample(
            self,
            test_number: RawNumberData,
            ann_forest: Ann,
            config: ANNTestConfig
    ) -> int:
        """
        Wykonuje predykcję dla pojedynczej próbki.

        Args:
            test_number: Dane testowe
            ann_forest: Model ANN
            config: Konfiguracja

        Returns:
            int: Przewidziana etykieta
        """
        test_number.binarize_data(config.pixel_normalization_rate, config.image_size)
        vector = VectorManager.create_vector_for_single_sample(test_number, config)
        return ann_forest.predict_label(vector.vector, self.num_classes)

    def _print_results(self, results: TestResult) -> None:
        """Wyświetla wyniki testowania."""
        print(f"Test zakończony:")
        print(f"  Poprawne predykcje: {results.correct_predictions}")
        print(f"  Niepoprawne predykcje: {results.incorrect_predictions}")
        print(f"  Dokładność: {results.accuracy:.2%}")
        print(f"  Precision: {results.precision:.4f}")
        print(f"  Recall: {results.recall:.4f}")
        print(f"  F1-Score: {results.f1_score:.4f}")
        print(f"  Czas wykonania: {results.execution_time:.3f}s")
