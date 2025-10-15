from dataclasses import dataclass
from enum import Enum
from typing import Any

from Testers.Shared.configs import BaseTestConfig, DimensionalityReductionAlgorithm


@dataclass
class SVMTestConfig:
    """Konfiguracja dla pojedynczego testu SVM"""

    # Parametry SVM
    C: float
    kernel: str  # 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
    degree: int  # dla 'poly'
    gamma: str  # 'scale', 'auto' lub float
    coef0: float  # dla 'poly' i 'sigmoid'
    shrinking: bool
    probability: bool
    random_state: int = 42

    # Parametry wektorów
    pixel_normalization_rate: float = 0.5  # Domyślnie 0.5, None dla metod statystycznych
    training_set_limit: int

    # Redukcja wymiarów
    dimensionality_reduction_algorithm: DimensionalityReductionAlgorithm = DimensionalityReductionAlgorithm.NONE
    dimensionality_reduction_n_components: int = 50  # Liczba komponentów do redukcji

    # Informacje o datasecie
    class_count: int
    image_size: int = 28  # Rozmiar obrazu (domyślnie 28x28)
    dataset_name: str = "Unknown"  # Nazwa datasetu dla raportów
    class_count: int
    image_size: int = 28  # Rozmiar obrazu (domyślnie 28x28)
    dataset_name: str = "Unknown"  # Nazwa datasetu dla raportów

    def __post_init__(self):
        """Walidacja konfiguracji po inicjalizacji"""
        if self.C <= 0:
            raise ValueError(f"C musi być > 0, otrzymano: {self.C}")
        if self.kernel not in ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']:
            raise ValueError(f"Kernel musi być jednym z: 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed', otrzymano: {self.kernel}")
        if self.degree < 1:
            raise ValueError(f"Degree musi być >= 1, otrzymano: {self.degree}")
        if isinstance(self.gamma, str) and self.gamma not in ['scale', 'auto']:
            raise ValueError(f"Gamma jako string musi być 'scale' lub 'auto', otrzymano: {self.gamma}")
        elif isinstance(self.gamma, (int, float)) and self.gamma <= 0:
            raise ValueError(f"Gamma jako liczba musi być > 0, otrzymano: {self.gamma}")
        if self.dimensionality_reduction_algorithm == DimensionalityReductionAlgorithm.FLOOD_FILL and not (0.0 <= self.pixel_normalization_rate <= 1.0):
            raise ValueError(
                f"Pixel normalization rate musi być w [0, 1] dla FLOOD_FILL, otrzymano: {self.pixel_normalization_rate}"
            )
        if self.dimensionality_reduction_n_components <= 0:
            raise ValueError(f"Dimensionality reduction n_components musi być > 0, otrzymano: {self.dimensionality_reduction_n_components}")
        if self.dimensionality_reduction_algorithm == DimensionalityReductionAlgorithm.LDA and self.dimensionality_reduction_n_components >= self.class_count:
            raise ValueError(f"Dla LDA n_components musi być < class_count ({self.class_count}), otrzymano: {self.dimensionality_reduction_n_components}")


class SVMTestConfigField(Enum):
    """Enum dla pól konfiguracji SVM"""
    C = "C"
    KERNEL = "kernel"
    DEGREE = "degree"
    GAMMA = "gamma"
    COEF0 = "coef0"
    SHRINKING = "shrinking"
    PROBABILITY = "probability"
    RANDOM_STATE = "random_state"
    PIXEL_NORMALIZATION_RATE = "pixel_normalization_rate"
    TRAINING_SET_LIMIT = "training_set_limit"
    DIMENSIONALITY_REDUCTION_ALGORITHM = "dimensionality_reduction_algorithm"
    DIMENSIONALITY_REDUCTION_N_COMPONENTS = "dimensionality_reduction_n_components"


@dataclass
class FieldConfig:
    """Konfiguracja dla pojedynczego pola w generatorze testów"""
    field_name: SVMTestConfigField
    start: Any
    stop: Any
    step: Any