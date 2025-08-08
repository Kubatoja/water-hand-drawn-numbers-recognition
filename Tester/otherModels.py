from dataclasses import dataclass, field
from typing import List

import numpy as np
from Tester.configs import ANNTestConfig


@dataclass
class TestResult:
    """Wyniki testowania modelu ANN."""
    config: ANNTestConfig
    training_time: float
    train_set_size: int
    test_set_size: int
    execution_time: float
    correct_predictions: int
    incorrect_predictions: int
    accuracy: float
    confusion_matrix: np.ndarray

    @property
    def total_predictions(self) -> int:
        """Łączna liczba predykcji."""
        return self.correct_predictions + self.incorrect_predictions

@dataclass
class RawNumberData:
    label: int
    pixels: np.ndarray = field(default_factory=lambda: np.array([]))

    def binarize_data(self, pixelNormalizationRate: float):
        self.pixels = np.where(self.pixels > pixelNormalizationRate, 1, 0).reshape(28, 28)

@dataclass
class VectorNumberData:
    label: int
    vector: List[float]
