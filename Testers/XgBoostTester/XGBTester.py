import time
from typing import List

import numpy as np
from xgboost import XGBClassifier

from Testers.Shared.models import TestResult, RawNumberData, VectorNumberData
from Testers.Shared.MetricsCalculator import MetricsCalculator
from Testers.Shared.VectorManager import VectorManager
from Testers.XgBoostTester.configs import XGBTestConfig


class XGBTester:
    """Klasa odpowiedzialna za testowanie modeli XGBoost"""

    def __init__(self, num_classes: int = 10):
        self.num_classes = num_classes
        self.metrics_calculator = MetricsCalculator()

    def test_model(
        self,
        model: XGBClassifier,
        test_vectors: List[VectorNumberData],
        config: XGBTestConfig
    ) -> TestResult:
        """
        Testuje model XGBoost na podanych danych testowych.

        Args:
            model: Wytrenowany model XGBoost
            test_vectors: Lista wektorów testowych
            config: Konfiguracja testu

        Returns:
            TestResult: Wyniki testowania
        """
        if not test_vectors:
            raise ValueError("Lista test_vectors nie może być pusta")

        start_time = time.perf_counter()

        # Przygotuj dane testowe
        X_test = np.array([vec.vector for vec in test_vectors])
        y_test = np.array([vec.label for vec in test_vectors])

        # Predykcja
        y_pred = model.predict(X_test)

        end_time = time.perf_counter()
        execution_time = end_time - start_time

        # Oblicz podstawowe metryki
        actual_labels = y_test.astype(int)
        predicted_labels = y_pred.astype(int)
        
        correct_predictions = np.sum(actual_labels == predicted_labels)
        total_predictions = len(test_vectors)
        incorrect_predictions = total_predictions - correct_predictions
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0

        # Oblicz rozszerzone metryki
        metrics = self.metrics_calculator.calculate_all_metrics(
            actual_labels, predicted_labels, self.num_classes
        )

        # Confusion matrix
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
        config: XGBTestConfig
    ) -> tuple[XGBClassifier, TestResult]:
        """
        Trenuje i testuje model XGBoost

        Args:
            training_vectors: Lista wektorów treningowych
            test_vectors: Lista wektorów testowych
            config: Konfiguracja testu

        Returns:
            Tuple(model, wyniki): Wytrenowany model i wyniki testowania
        """
        # Przygotuj dane treningowe
        X_train = np.array([vec.vector for vec in training_vectors])
        y_train = np.array([vec.label for vec in training_vectors])

        # Trenowanie
        print("Training XGBoost model...")
        train_start = time.perf_counter()

        model = XGBClassifier(
            learning_rate=config.learning_rate,
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            min_child_weight=config.min_child_weight,
            gamma=config.gamma,
            subsample=config.subsample,
            colsample_bytree=config.colsample_bytree,
            reg_lambda=config.reg_lambda,
            reg_alpha=config.reg_alpha,
            objective='multi:softmax',
            num_class=self.num_classes,
            random_state=42,
            n_jobs=-1,  # Użyj wszystkich dostępnych rdzeni
            verbosity=0  # Wyłącz verbose output
        )

        model.fit(X_train, y_train)

        train_end = time.perf_counter()
        training_time = train_end - train_start

        print(f"📊 Training completed in {training_time:.3f}s")

        # Testowanie
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
