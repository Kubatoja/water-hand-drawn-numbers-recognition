import time
from typing import List, Dict, Optional

import numpy as np
from sklearn.model_selection import StratifiedKFold

from pytabkit import TabR_S_D_Classifier
from Testers.Shared.models import TestResult, VectorNumberData
from Testers.Shared.MetricsCalculator import MetricsCalculator
from .configs import TabRTestConfig


class TabRTester:
    """Klasa odpowiedzialna za testowanie modeli TabR"""

    def __init__(self, num_classes: int = 10):
        self.num_classes = num_classes
        self.metrics_calculator = MetricsCalculator()

    def test_model(
        self,
        model: TabR_S_D_Classifier,
        test_vectors: List[VectorNumberData],
        config: TabRTestConfig
    ) -> TestResult:
        """
        Testuje model TabR na podanych danych testowych.

        Args:
            model: Wytrenowany model TabR
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

        # TabR_S_D_Classifier.predict() zwraca bezpośrednio etykiety klas
        # (zachowanie identyczne z SVC.predict), bez potrzeby argmax
        y_pred = model.predict(X_test)

        end_time = time.perf_counter()
        execution_time = end_time - start_time

        actual_labels = y_test.astype(int)
        predicted_labels = y_pred.astype(int)

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

    def _build_model(self, config: TabRTestConfig) -> TabR_S_D_Classifier:
        """
        Tworzy instancję modelu TabR_S_D_Classifier na podstawie konfiguracji.
        Przekazywane są wyłącznie podstawowe parametry (n_epochs, batch_size,
        learning_rate); pozostałe hiperparametry zachowują wartości domyślne
        zdefiniowane przez bibliotekę pytabkit.

        Args:
            config: Konfiguracja TabR

        Returns:
            Nowa instancja TabR_S_D_Classifier
        """
        # Resolve device automatically if needed
        # Force device to 'cpu' for TabR
        return TabR_S_D_Classifier(
            n_epochs=config.n_epochs,
            batch_size=config.batch_size,
            optimizer={"type": "Adam", "lr": config.learning_rate},
            random_state=config.random_state,
            device="cpu",
            n_cv=1,
            n_refit=0,
            verbosity=config.verbose,
        )

    def _perform_cross_validation(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        config: TabRTestConfig,
        n_folds: int = 5
    ) -> tuple[Dict[str, float], list[TestResult]]:
        """
        Wykonuje stratified k-fold cross-validation na training set.

        TabR_S_D_Classifier posiada wbudowany mechanizm n_cv (bagging CV),
        jednak tutaj realizujemy zewnętrzne CV ręcznie – analogicznie
        do GRANDETester – aby zachować spójność z pozostałymi testerami
        i unikać podwójnego CV.

        Args:
            X_train: Macierz cech treningowych
            y_train: Wektor etykiet treningowych
            config: Konfiguracja TabR
            n_folds: Liczba foldów dla CV

        Returns:
            Dict ze średnimi i odchyleniami standardowymi metryk CV
        """
        print(f"  Performing {n_folds}-fold stratified cross-validation...")

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        fold_accuracies = []
        fold_precisions = []
        fold_recalls = []
        fold_f1s = []
        fold_results = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

            fold_model = self._build_model(config)
            fold_model.fit(X_fold_train, y_fold_train)

            y_fold_pred = fold_model.predict(X_fold_val).astype(int)

            fold_metrics = self.metrics_calculator.calculate_all_metrics(
                y_fold_val.astype(int), y_fold_pred, self.num_classes
            )

            acc = np.sum(y_fold_val.astype(int) == y_fold_pred) / len(y_fold_val)
            fold_accuracies.append(acc)
            fold_precisions.append(fold_metrics['precision'])
            fold_recalls.append(fold_metrics['recall'])
            fold_f1s.append(fold_metrics['f1_score'])

            confusion_matrix = self.metrics_calculator.calculate_confusion_matrix(
                y_fold_val.astype(int), y_fold_pred, self.num_classes
            )

            fold_results.append(TestResult(
                execution_time=None,
                correct_predictions=int(np.sum(y_fold_val.astype(int) == y_fold_pred)),
                incorrect_predictions=int(len(y_fold_val) - np.sum(y_fold_val.astype(int) == y_fold_pred)),
                accuracy=acc,
                confusion_matrix=confusion_matrix,
                precision=fold_metrics['precision'],
                recall=fold_metrics['recall'],
                f1_score=fold_metrics['f1_score'],
                per_class_precision=fold_metrics.get('per_class_precision', np.array([])),
                per_class_recall=fold_metrics.get('per_class_recall', np.array([])),
                per_class_f1=fold_metrics.get('per_class_f1', np.array([])),
                config=config,
                training_time=None,
                train_set_size=len(X_fold_train),
                test_set_size=len(X_fold_val),
                fold_id=fold_idx + 1
            ))

            print(f"  Fold {fold_idx + 1}/{n_folds}: Accuracy={acc:.4f}, F1={fold_metrics['f1_score']:.4f}")

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
        print(f"              F1-Score  = {cv_scores['cv_f1_mean']:.4f} ± {cv_scores['cv_f1_std']:.4f}")

        return cv_scores, fold_results

    def train_and_test(
        self,
        training_vectors: List[VectorNumberData],
        test_vectors: List[VectorNumberData],
        config: TabRTestConfig,
        use_cross_validation: bool = True,
        cv_n_folds: int = 5
    ) -> tuple[TabR_S_D_Classifier, TestResult, list[TestResult]]:
        """
        Trenuje i testuje model TabR.

        Args:
            training_vectors: Lista wektorów treningowych
            test_vectors: Lista wektorów testowych
            config: Konfiguracja testu
            use_cross_validation: Czy wykonać cross-validation
            cv_n_folds: Liczba foldów dla CV

        Returns:
            Tuple(model, wyniki): Wytrenowany model i wyniki testowania
        """
        X_train = np.array([vec.vector for vec in training_vectors])
        y_train = np.array([vec.label for vec in training_vectors])

        # Cross-validation (jeśli włączone)
        cv_scores = None
        fold_results = None
        if use_cross_validation:
            cv_scores, fold_results = self._perform_cross_validation(
                X_train, y_train, config, n_folds=cv_n_folds
            )

        # Trenowanie na pełnym training set
        print("Training TabR model on full training set...")
        train_start = time.perf_counter()

        model = self._build_model(config)

        # TabR_S_D_Classifier jest w pełni sklearn-kompatybilny:
        # wewnętrzny val split do early stopping jest obsługiwany automatycznie
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

        return model, result, fold_results

    def _print_results(self, results: TestResult) -> None:
        """Wyświetla wyniki testowania"""
        print(f"Test zakończony:")
        print(f"  Poprawne predykcje:    {results.correct_predictions}")
        print(f"  Niepoprawne predykcje: {results.incorrect_predictions}")
        print(f"  Dokładność:            {results.accuracy:.2%}")
        print(f"  Precision:             {results.precision:.4f}")
        print(f"  Recall:                {results.recall:.4f}")
        print(f"  F1-Score:              {results.f1_score:.4f}")
        print(f"  Czas wykonania:        {results.execution_time:.3f}s")