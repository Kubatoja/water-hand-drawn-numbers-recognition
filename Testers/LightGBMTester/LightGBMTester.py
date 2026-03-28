import time
from typing import List, Dict

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

from Testers.Shared.models import TestResult, RawNumberData, VectorNumberData
from Testers.Shared.MetricsCalculator import MetricsCalculator
from Testers.Shared.VectorManager import VectorManager
from Testers.LightGBMTester.configs import LightGBMTestConfig


class LightGBMTester:
    """Klasa odpowiedzialna za testowanie modeli LightGBM"""

    def __init__(self, num_classes: int = 10):
        self.num_classes = num_classes
        self.metrics_calculator = MetricsCalculator()

    def test_model(
        self,
        model: LGBMClassifier,
        test_vectors: List[VectorNumberData],
        config: LightGBMTestConfig
    ) -> TestResult:
        if not test_vectors:
            raise ValueError("Lista test_vectors nie może być pusta")

        start_time = time.perf_counter()
        X_test = np.array([vec.vector for vec in test_vectors])
        y_test = np.array([vec.label for vec in test_vectors])

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

    def _perform_cross_validation(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        config: LightGBMTestConfig,
        n_folds: int = 5
    ) -> tuple[Dict[str, float], list[TestResult]]:
        print(f"  Performing {n_folds}-fold stratified cross-validation...")

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        fold_results = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

            fold_model = LGBMClassifier(
                learning_rate=config.learning_rate,
                n_estimators=config.n_estimators,
                max_depth=config.max_depth,
                num_leaves=config.num_leaves,
                subsample=config.subsample,
                colsample_bytree=config.colsample_bytree,
                random_state=config.random_state,
                n_jobs=config.n_jobs,
                objective='multiclass',
                num_class=self.num_classes,
                verbosity=-1
            )
            fold_model.fit(X_fold_train, y_fold_train)

            y_pred = fold_model.predict(X_fold_val)
            actual_labels = y_fold_val.astype(int)
            predicted_labels = y_pred.astype(int)
            correct_predictions = np.sum(actual_labels == predicted_labels)
            total_predictions = len(y_fold_val)
            incorrect_predictions = total_predictions - correct_predictions
            acc = correct_predictions / total_predictions if total_predictions > 0 else 0.0

            metrics = self.metrics_calculator.calculate_all_metrics(actual_labels, predicted_labels, self.num_classes)
            confusion_matrix = self.metrics_calculator.calculate_confusion_matrix(actual_labels, predicted_labels, self.num_classes)

            fold_results.append(TestResult(
                execution_time=None,
                correct_predictions=int(correct_predictions),
                incorrect_predictions=int(incorrect_predictions),
                accuracy=acc,
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
                fold_id=fold_idx + 1
            ))

        model = LGBMClassifier(
            learning_rate=config.learning_rate,
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            num_leaves=config.num_leaves,
            subsample=config.subsample,
            colsample_bytree=config.colsample_bytree,
            random_state=config.random_state,
            n_jobs=config.n_jobs,
            objective='multiclass',
            num_class=self.num_classes,
            verbosity=-1
        )

        scoring = {
            'accuracy': 'accuracy',
            'precision_macro': 'precision_macro',
            'recall_macro': 'recall_macro',
            'f1_macro': 'f1_macro'
        }

        cv_results = cross_validate(
            model, X_train, y_train,
            cv=skf,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False
        )

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
        return cv_scores, fold_results

    def train_and_test(
        self,
        training_vectors: List[VectorNumberData],
        test_vectors: List[VectorNumberData],
        config: LightGBMTestConfig,
        use_cross_validation: bool = True,
        cv_n_folds: int = 5
    ) -> tuple[LGBMClassifier, TestResult, list[TestResult]]:
        X_train = np.array([vec.vector for vec in training_vectors])
        y_train = np.array([vec.label for vec in training_vectors])

        cv_scores = None
        fold_results = None
        if use_cross_validation:
            cv_scores, fold_results = self._perform_cross_validation(
                X_train, y_train, config, n_folds=cv_n_folds
            )

        print("Training LightGBM model on full training set...")
        train_start = time.perf_counter()

        model = LGBMClassifier(
            learning_rate=config.learning_rate,
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            num_leaves=config.num_leaves,
            subsample=config.subsample,
            colsample_bytree=config.colsample_bytree,
            random_state=config.random_state,
            n_jobs=config.n_jobs,
            objective='multiclass',
            num_class=self.num_classes,
            verbosity=-1
        )

        model.fit(X_train, y_train)

        train_end = time.perf_counter()
        training_time = train_end - train_start
        print(f"📊 Training completed in {training_time:.3f}s")

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
        print(f"Test zakończony:")
        print(f"  Poprawne predykcje: {results.correct_predictions}")
        print(f"  Niepoprawne predykcje: {results.incorrect_predictions}")
        print(f"  Dokładność: {results.accuracy:.2%}")
        print(f"  Precision: {results.precision:.4f}")
        print(f"  Recall: {results.recall:.4f}")
        print(f"  F1-Score: {results.f1_score:.4f}")
