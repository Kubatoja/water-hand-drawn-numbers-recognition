import numpy as np
from sklearn.metrics import precision_recall_fscore_support, precision_score, recall_score, f1_score


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
        # Macro-averaged metryki (średnia dla wszystkich klas)
        precision_macro = precision_score(
            actual_labels, predicted_labels, 
            average='macro', zero_division=0
        )
        recall_macro = recall_score(
            actual_labels, predicted_labels, 
            average='macro', zero_division=0
        )
        f1_macro = f1_score(
            actual_labels, predicted_labels, 
            average='macro', zero_division=0
        )
        
        # Per-class metryki
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            actual_labels, predicted_labels,
            labels=list(range(num_classes)),
            average=None,
            zero_division=0
        )
        
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
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
        
        for actual, predicted in zip(actual_labels, predicted_labels):
            confusion_matrix[int(actual)][int(predicted)] += 1
            
        return confusion_matrix
    
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
