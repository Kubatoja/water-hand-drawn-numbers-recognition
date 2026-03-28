import time
from typing import List

import numpy as np

from Testers.Shared.models import TestResult, VectorNumberData
from Testers.Shared.MetricsCalculator import MetricsCalculator
from Testers.HyperFastTester.configs import HyperFastTestConfig
from sklearn.model_selection import StratifiedKFold


def _resolve_device(device: str) -> str:
    """Wymusza użycie CPU niezależnie od ustawień."""
    return "cuda" if device == "cuda" else "cpu"


class HyperFastTester:
    """Klasa odpowiedzialna za testowanie modeli HyperFast"""

    def __init__(self, num_classes: int = 10):
        self.num_classes = num_classes
        self.metrics_calculator = MetricsCalculator()

    def test_model(
        self,
        model,
        test_vectors: List[VectorNumberData],
        config: HyperFastTestConfig
    ) -> TestResult:
        """
        Testuje model HyperFast na podanych danych testowych.

        Args:
            model: Wytrenowany model HyperFast
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
        config: HyperFastTestConfig,
        use_cross_validation: bool = True,
        cv_n_folds: int = 5
    ) -> tuple:
        """
        Trenuje i testuje model HyperFast.

        Args:
            training_vectors: Lista wektorów treningowych
            test_vectors: Lista wektorów testowych
            config: Konfiguracja testu
            use_cross_validation: Nieużywane dla HyperFast (zachowane dla kompatybilności API)
            cv_n_folds: Nieużywane dla HyperFast (zachowane dla kompatybilności API)

        Returns:
            Tuple(model, wyniki): Wytrenowany model i wyniki testowania
        """
        from hyperfast import HyperFastClassifier

        X_train = np.array([vec.vector for vec in training_vectors])
        y_train = np.array([vec.label for vec in training_vectors])

        device = _resolve_device(config.device)
        print(f"Training HyperFast model on device='{device}'...")

        train_start = time.perf_counter()

        model = HyperFastClassifier(
            device=device,
            n_ensemble=config.n_ensemble,
            batch_size=config.batch_size,
            nn_bias=config.nn_bias,
            optimization=config.optimization,
            optimize_steps=config.optimize_steps,
            seed=config.random_state,
        )

        fold_results = []
        cv_scores = None

        if use_cross_validation:
            fold_accuracies = []
            fold_precisions = []
            fold_recalls = []
            fold_f1s = []

            skf = StratifiedKFold(n_splits=cv_n_folds, shuffle=True, random_state=42)
            for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), start=1):
                X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
                y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

                fold_model = HyperFastClassifier(
                    device=device,
                    n_ensemble=config.n_ensemble,
                    batch_size=config.batch_size,
                    nn_bias=config.nn_bias,
                    optimization=config.optimization,
                    optimize_steps=config.optimize_steps,
                    seed=config.random_state,
                )

                fold_model.fit(X_fold_train, y_fold_train)
                y_fold_pred = fold_model.predict(X_fold_val)

                actual_labels = y_fold_val.astype(int)
                predicted_labels = np.array(y_fold_pred).astype(int)

                correct_predictions = int(np.sum(actual_labels == predicted_labels))
                total_predictions = len(y_fold_val)
                incorrect_predictions = int(total_predictions - correct_predictions)
                accuracy = float(correct_predictions / total_predictions) if total_predictions > 0 else 0.0

                metrics = self.metrics_calculator.calculate_all_metrics(actual_labels, predicted_labels, self.num_classes)
                confusion_matrix = self.metrics_calculator.calculate_confusion_matrix(actual_labels, predicted_labels, self.num_classes)

                fold_results.append(TestResult(
                    execution_time=None,
                    correct_predictions=correct_predictions,
                    incorrect_predictions=incorrect_predictions,
                    accuracy=accuracy,
                    confusion_matrix=confusion_matrix,
                    precision=metrics['precision'],
                    recall=metrics['recall'],
                    f1_score=metrics['f1_score'],
                    per_class_precision=metrics['per_class_precision'],
                    per_class_recall=metrics['per_class_recall'],
                    per_class_f1=metrics['per_class_f1'],
                    config=config,
                    training_time=None,
                    train_set_size=len(X_fold_train),
                    test_set_size=len(X_fold_val),
                    fold_id=fold_idx
                ))

                fold_accuracies.append(accuracy)
                fold_precisions.append(metrics['precision'])
                fold_recalls.append(metrics['recall'])
                fold_f1s.append(metrics['f1_score'])

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

        model.fit(X_train, y_train)
        train_end = time.perf_counter()
        training_time = train_end - train_start
        print(f"Training completed in {training_time:.3f}s")

        print("Running test evaluation...")
        result = self.test_model(model, test_vectors, config)
        result.training_time = training_time

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
        print(f"  Poprawne predykcje: {results.correct_predictions}")
        print(f"  Niepoprawne predykcje: {results.incorrect_predictions}")
        print(f"  Dokładność: {results.accuracy:.2%}")
        print(f"  Precision: {results.precision:.4f}")
        print(f"  Recall: {results.recall:.4f}")
        print(f"  F1 Score: {results.f1_score:.4f}")
