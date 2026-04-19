import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix as sklearn_confusion_matrix


class MetricsCalculator:
    """Kalkuluje różne metryki dla wyników klasyfikacji"""

    @staticmethod
    def calculate_all_metrics(
        actual_labels: np.ndarray,
        predicted_labels: np.ndarray,
        num_classes: int
    ) -> dict:
        """
        Kalkuluje wszystkie metryki dla wyników klasyfikacji
        
        Args:
            actual_labels: Rzeczywiste etykiety
            predicted_labels: Przewidywane etykiety
            num_classes: Liczba klas
            
        Returns:
            Słownik z metrykami
        """
        # Jeden przebieg po danych: metryki per-class, a macro wyliczamy jako średnią.
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            actual_labels, predicted_labels,
            labels=list(range(num_classes)),
            average=None,
            zero_division=0
        )

        precision_macro = float(np.mean(precision_per_class))
        recall_macro = float(np.mean(recall_per_class))
        f1_macro = float(np.mean(f1_per_class))
        
        return {
            'precision': precision_macro,
            'recall': recall_macro,
            'f1_score': f1_macro,
            'per_class_precision': precision_per_class,
            'per_class_recall': recall_per_class,
            'per_class_f1': f1_per_class
        }
    
    @staticmethod
    def calculate_confusion_matrix(
        actual_labels: np.ndarray,
        predicted_labels: np.ndarray,
        num_classes: int
    ) -> np.ndarray:
        """
        Tworzy confusion matrix
        
        Args:
            actual_labels: Rzeczywiste etykiety
            predicted_labels: Przewidywane etykiety
            num_classes: Liczba klas
            
        Returns:
            Confusion matrix jako numpy array
        """
        return sklearn_confusion_matrix(
            actual_labels,
            predicted_labels,
            labels=list(range(num_classes))
        )

    @staticmethod
    def normalize_labels(labels: np.ndarray) -> np.ndarray:
        """
        Normalizuje etykiety do formatu 1D wektora klas.

        Obsługuje wyniki predykcji w postaci:
        - 1D wektora klas
        - kolumnowego wektora kształtu (n_samples, 1)
        - macierzy prawdopodobieństw lub one-hot kształtu (n_samples, n_classes)
        """
        arr = np.asarray(labels)

        if arr.ndim > 1:
            if arr.shape[1] == 1:
                arr = arr.ravel()
            else:
                arr = np.argmax(arr, axis=1)

        return arr.astype(int)

    @staticmethod
    def prepare_labels(actual_labels: np.ndarray, predicted_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Przygotowuje etykiety do obliczeń metryk.

        Zwraca 1D wektory etykiet i weryfikuje ich kształt.
        """
        actual = MetricsCalculator.normalize_labels(actual_labels)
        predicted = MetricsCalculator.normalize_labels(predicted_labels)

        if actual.shape != predicted.shape:
            raise ValueError(
                f"Shape mismatch between actual labels {actual.shape} "
                f"and predicted labels {predicted.shape}"
            )

        return actual, predicted

    @staticmethod
    def print_metrics(metrics: dict, detailed: bool = True):
        """
        Wyświetla metryki w czytelny sposób
        
        Args:
            metrics: Słownik z metrykami
            detailed: Czy wyświetlać szczegółowe metryki per-class
        """
        print(f"Accuracy:  {metrics.get('accuracy', 0):.4f}")
        print(f"Precision: {metrics.get('precision', 0):.4f}")
        print(f"Recall:    {metrics.get('recall', 0):.4f}")
        print(f"F1-Score:  {metrics.get('f1_score', 0):.4f}")
        
        if detailed and 'per_class_precision' in metrics:
            print("\nPer-class metrics:")
            for i, (p, r, f1) in enumerate(zip(
                metrics['per_class_precision'],
                metrics['per_class_recall'],
                metrics['per_class_f1']
            )):
                print(f"  Class {i}: P={p:.3f}, R={r:.3f}, F1={f1:.3f}")
