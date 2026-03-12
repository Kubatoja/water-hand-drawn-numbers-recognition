import time
from typing import List, Dict, Optional

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from Testers.Shared.models import TestResult, VectorNumberData
from Testers.Shared.MetricsCalculator import MetricsCalculator
from Testers.TabICLTester.configs import TabICLTestConfig


def _resolve_device(device: str) -> str:
    """Wykrywa dostępne urządzenie."""
    if device == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return device


class TabICLTester:
    """Klasa odpowiedzialna za testowanie modeli TabICLv2"""

    def __init__(self, num_classes: int = 10):
        self.num_classes = num_classes
        self.metrics_calculator = MetricsCalculator()

    def test_model(
        self,
        model,
        test_vectors: List[VectorNumberData],
        config: TabICLTestConfig
    ) -> TestResult:
        """
        Testuje model TabICL na podanych danych testowych.

        Args:
            model: Wytrenowany model TabICL (z zapisanym kontekstem treningowym)
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

    def _perform_cross_validation_manual(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        config: TabICLTestConfig,
        n_folds: int = 5
    ) -> Dict[str, float]:
        """
        Ręczna implementacja stratified k-fold cross-validation dla TabICL.
        
        Args:
            X_train: Macierz cech treningowych
            y_train: Wektor etykiet treningowych
            config: Konfiguracja TabICL
            n_folds: Liczba foldów dla CV
            
        Returns:
            Dict ze średnimi i odchyleniami standardowymi metryk CV
        """
        from tabicl import TabICLClassifier
        
        print(f"  Performing {n_folds}-fold stratified cross-validation (manual loop)...")
        
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        fold_accuracies = []
        fold_precisions = []
        fold_recalls = []
        fold_f1s = []
        
        device = _resolve_device(config.device)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
            
            # Stwórz i trenuj model
            model = TabICLClassifier(
                n_estimators=config.n_estimators,
                softmax_temperature=config.softmax_temperature,
                outlier_threshold=config.outlier_threshold,
                support_many_classes=True,
                device=device,
                random_state=config.random_state,
            )
            
            model.fit(X_fold_train, y_fold_train)
            y_pred = model.predict(X_fold_val)
            
            # Oblicz metryki
            fold_accuracies.append(accuracy_score(y_fold_val, y_pred))
            fold_precisions.append(precision_score(y_fold_val, y_pred, average='macro', zero_division=0))
            fold_recalls.append(recall_score(y_fold_val, y_pred, average='macro', zero_division=0))
            fold_f1s.append(f1_score(y_fold_val, y_pred, average='macro', zero_division=0))
            
            print(f"    Fold {fold}/{n_folds}: Accuracy={fold_accuracies[-1]:.4f}")
        
        cv_scores = {
            'cv_accuracy_mean': float(np.mean(fold_accuracies)),
            'cv_accuracy_std': float(np.std(fold_accuracies)),
            'cv_precision_mean': float(np.mean(fold_precisions)),
            'cv_precision_std': float(np.std(fold_precisions)),
            'cv_recall_mean': float(np.mean(fold_recalls)),
            'cv_recall_std': float(np.std(fold_recalls)),
            'cv_f1_mean': float(np.mean(fold_f1s)),
            'cv_f1_std': float(np.std(fold_f1s)),
        }
        
        print(f"  CV Results: Accuracy = {cv_scores['cv_accuracy_mean']:.4f} ± {cv_scores['cv_accuracy_std']:.4f}")
        print(f"              F1-Score = {cv_scores['cv_f1_mean']:.4f} ± {cv_scores['cv_f1_std']:.4f}")
        
        return cv_scores

    def train_and_test(
        self,
        training_vectors: List[VectorNumberData],
        test_vectors: List[VectorNumberData],
        config: TabICLTestConfig,
        use_cross_validation: bool = True,
        cv_n_folds: int = 5
    ) -> tuple:
        """
        Trenuje i testuje model TabICLv2 (fit + predict w jednym forward pass).

        Args:
            training_vectors: Lista wektorów treningowych
            test_vectors: Lista wektorów testowych
            config: Konfiguracja testu
            use_cross_validation: Czy wykonać cross-validation
            cv_n_folds: Liczba foldów dla CV

        Returns:
            Tuple(model, wyniki): Model z wczytanym kontekstem i wyniki testowania
        """
        from tabicl import TabICLClassifier

        X_train = np.array([vec.vector for vec in training_vectors])
        y_train = np.array([vec.label for vec in training_vectors])

        # Cross-validation (jeśli włączone)
        cv_scores = None
        if use_cross_validation:
            cv_scores = self._perform_cross_validation_manual(
                X_train, y_train, config, n_folds=cv_n_folds
            )

        device = _resolve_device(config.device)
        print(f"Training TabICLv2 model on full training set (device='{device}')...")

        train_start = time.perf_counter()

        model = TabICLClassifier(
            n_estimators=config.n_estimators,
            softmax_temperature=config.softmax_temperature,
            outlier_threshold=config.outlier_threshold,
            support_many_classes=True,
            device=device,
            random_state=config.random_state,
        )

        # fit() w TabICL jest tanie obliczeniowo – właściwy ICL dzieje się w predict()
        model.fit(X_train, y_train)

        train_end = time.perf_counter()
        training_time = train_end - train_start
        print(f"📊 Fit completed in {training_time:.3f}s")

        print("Running test evaluation (in-context learning forward pass)...")
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
