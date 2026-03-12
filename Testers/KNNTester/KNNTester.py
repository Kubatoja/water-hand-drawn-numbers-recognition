import time
from typing import List, Dict, Optional

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

from Testers.Shared.models import TestResult, RawNumberData, VectorNumberData
from Testers.Shared.MetricsCalculator import MetricsCalculator
from Testers.Shared.VectorManager import VectorManager
from Testers.KNNTester.configs import KNNTestConfig


class KNNTester:
    """Klasa odpowiedzialna za testowanie modeli KNN"""

    def __init__(self, num_classes: int = 10):
        self.num_classes = num_classes
        self.metrics_calculator = MetricsCalculator()

    def test_model(
        self,
        model: KNeighborsClassifier,
        test_vectors: List[VectorNumberData],
        config: KNNTestConfig
    ) -> TestResult:
        """
        Testuje model KNN na podanych danych testowych.

        Args:
            model: Wytrenowany model KNN
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

    def _perform_cross_validation(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        config: KNNTestConfig,
        n_folds: int = 5
    ) -> Dict[str, float]:
        """
        Wykonuje stratified k-fold cross-validation na training set.
        
        Args:
            X_train: Macierz cech treningowych
            y_train: Wektor etykiet treningowych
            config: Konfiguracja KNN
            n_folds: Liczba foldów dla CV
            
        Returns:
            Dict ze średnimi i odchyleniami standardowymi metryk CV
        """
        print(f"  Performing {n_folds}-fold stratified cross-validation...")
        
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        # Stwórz model do CV
        model = KNeighborsClassifier(
            n_neighbors=config.n_neighbors,
            weights=config.weights,
            algorithm=config.algorithm,
            leaf_size=config.leaf_size,
            p=config.p,
            metric=config.metric,
            n_jobs=-1
        )
        
        # Scoring metrics
        scoring = {
            'accuracy': 'accuracy',
            'precision_macro': 'precision_macro',
            'recall_macro': 'recall_macro',
            'f1_macro': 'f1_macro'
        }
        
        # Wykonaj CV
        cv_results = cross_validate(
            model, X_train, y_train,
            cv=skf,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False
        )
        
        # Oblicz średnie i std
        cv_scores = {
            'cv_accuracy_mean': float(cv_results['test_accuracy'].mean()),
            'cv_accuracy_std': float(cv_results['test_accuracy'].std()),
            'cv_precision_mean': float(cv_results['test_precision_macro'].mean()),
            'cv_precision_std': float(cv_results['test_precision_macro'].std()),
            'cv_recall_mean': float(cv_results['test_recall_macro'].mean()),
            'cv_recall_std': float(cv_results['test_recall_macro'].std()),
            'cv_f1_mean': float(cv_results['test_f1_macro'].mean()),
            'cv_f1_std': float(cv_results['test_f1_macro'].std()),
        }
        
        print(f"  CV Results: Accuracy = {cv_scores['cv_accuracy_mean']:.4f} ± {cv_scores['cv_accuracy_std']:.4f}")
        print(f"              F1-Score = {cv_scores['cv_f1_mean']:.4f} ± {cv_scores['cv_f1_std']:.4f}")
        
        return cv_scores

    def train_and_test(
        self,
        training_vectors: List[VectorNumberData],
        test_vectors: List[VectorNumberData],
        config: KNNTestConfig,
        use_cross_validation: bool = True,
        cv_n_folds: int = 5
    ) -> tuple[KNeighborsClassifier, TestResult]:
        """
        Trenuje i testuje model KNN

        Args:
            training_vectors: Lista wektorów treningowych
            test_vectors: Lista wektorów testowych
            config: Konfiguracja testu
            use_cross_validation: Czy wykonać cross-validation
            cv_n_folds: Liczba foldów dla CV

        Returns:
            Tuple(model, wyniki): Wytrenowany model i wyniki testowania
        """
        # Przygotuj dane treningowe
        X_train = np.array([vec.vector for vec in training_vectors])
        y_train = np.array([vec.label for vec in training_vectors])

        # Cross-validation (jeśli włączone)
        cv_scores = None
        if use_cross_validation:
            cv_scores = self._perform_cross_validation(
                X_train, y_train, config, n_folds=cv_n_folds
            )

        # Trenowanie na pełnym training set
        print("Training KNN model on full training set...")
        train_start = time.perf_counter()

        model = KNeighborsClassifier(
            n_neighbors=config.n_neighbors,
            weights=config.weights,
            algorithm=config.algorithm,
            leaf_size=config.leaf_size,
            p=config.p,
            metric=config.metric,
            n_jobs=-1  # Użyj wszystkich dostępnych rdzeni
        )

        model.fit(X_train, y_train)

        train_end = time.perf_counter()
        training_time = train_end - train_start

        print(f"📊 Training completed in {training_time:.3f}s")

        # Testowanie na test set
        print("Running test evaluation...")
        result = self.test_model(model, test_vectors, config)
        result.training_time = training_time
        
        # Dodaj CV scores do wyniku
        if cv_scores:
            result.cv_accuracy_mean = cv_scores['cv_accuracy_mean']
            result.cv_accuracy_std = cv_scores['cv_accuracy_std']
            result.cv_precision_mean = cv_scores['cv_precision_mean']
            result.cv_precision_std = cv_scores['cv_precision_std']
            result.cv_recall_mean = cv_scores['cv_recall_mean']
            result.cv_recall_std = cv_scores['cv_recall_std']
            result.cv_f1_mean = cv_scores['cv_f1_mean']
            result.cv_f1_std = cv_scores['cv_f1_std']

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