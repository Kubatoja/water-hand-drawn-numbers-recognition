from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol

import numpy as np


class BaseTestConfig(Protocol):
    """Protokół definiujący interfejs dla konfiguracji testów"""
    pixel_normalization_rate: float
    num_segments: int
    training_set_limit: int
    class_count: int


@dataclass
class TestResult:
    """Wyniki testowania modelu - uniwersalne dla różnych algorytmów."""
    config: Any  # BaseTestConfig - ale używamy Any dla elastyczności
    training_time: float
    train_set_size: int
    test_set_size: int
    execution_time: float
    correct_predictions: int
    incorrect_predictions: int
    accuracy: float
    confusion_matrix: np.ndarray
    
    # Rozszerzone metryki
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    
    # Metryki per-class (opcjonalne)
    per_class_precision: np.ndarray = field(default_factory=lambda: np.array([]))
    per_class_recall: np.ndarray = field(default_factory=lambda: np.array([]))
    per_class_f1: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def total_predictions(self) -> int:
        """Łączna liczba predykcji."""
        return self.correct_predictions + self.incorrect_predictions


@dataclass
class RawNumberData:
    """Surowe dane obrazu cyfry"""
    label: int
    pixels: np.ndarray = field(default_factory=lambda: np.array([]))

    def binarize_data(self, pixel_normalization_rate: float, image_size: int = 28):
        """
        Binaryzuje piksele na podstawie progu normalizacji
        
        Args:
            pixel_normalization_rate: Próg binaryzacji (0.0-1.0)
            image_size: Rozmiar obrazu (domyślnie 28 dla 28x28)
        """
        self.pixels = np.where(
            self.pixels > pixel_normalization_rate, 1, 0
        ).reshape(image_size, image_size)


@dataclass
class VectorNumberData:
    """Dane w formie wektora cech"""
    label: int
    vector: List[float]
