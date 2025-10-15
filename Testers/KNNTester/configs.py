from dataclasses import dataclass
from enum import Enum
from typing import Any

from Testers.Shared.configs import BaseTestConfig, DimensionalityReductionAlgorithm


@dataclass
class KNNTestConfig:
    """Konfiguracja dla pojedynczego testu KNN"""

    # Parametry KNN
    n_neighbors: int
    weights: str  # 'uniform', 'distance'
    algorithm: str  # 'auto', 'ball_tree', 'kd_tree', 'brute'
    leaf_size: int
    p: int  # 1 for manhattan, 2 for euclidean
    metric: str

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

    def __post_init__(self):
        """Walidacja konfiguracji po inicjalizacji"""
        if self.n_neighbors <= 0:
            raise ValueError(f"N neighbors musi być > 0, otrzymano: {self.n_neighbors}")
        if self.weights not in ['uniform', 'distance']:
            raise ValueError(f"Weights musi być 'uniform' lub 'distance', otrzymano: {self.weights}")
        if self.algorithm not in ['auto', 'ball_tree', 'kd_tree', 'brute']:
            raise ValueError(f"Algorithm musi być jednym z: 'auto', 'ball_tree', 'kd_tree', 'brute', otrzymano: {self.algorithm}")
        if self.leaf_size <= 0:
            raise ValueError(f"Leaf size musi być > 0, otrzymano: {self.leaf_size}")
        if self.p not in [1, 2]:
            raise ValueError(f"P musi być 1 lub 2, otrzymano: {self.p}")
        if self.dimensionality_reduction_algorithm == DimensionalityReductionAlgorithm.FLOOD_FILL and not (0.0 <= self.pixel_normalization_rate <= 1.0):
            raise ValueError(
                f"Pixel normalization rate musi być w [0, 1] dla FLOOD_FILL, otrzymano: {self.pixel_normalization_rate}"
            )
        if self.dimensionality_reduction_n_components <= 0:
            raise ValueError(f"Dimensionality reduction n_components musi być > 0, otrzymano: {self.dimensionality_reduction_n_components}")
        if self.dimensionality_reduction_algorithm == DimensionalityReductionAlgorithm.LDA and self.dimensionality_reduction_n_components >= self.class_count:
            raise ValueError(f"Dla LDA n_components musi być < class_count ({self.class_count}), otrzymano: {self.dimensionality_reduction_n_components}")


class KNNTestConfigField(Enum):
    """Enum dla pól konfiguracji KNN"""
    N_NEIGHBORS = "n_neighbors"
    WEIGHTS = "weights"
    ALGORITHM = "algorithm"
    LEAF_SIZE = "leaf_size"
    P = "p"
    METRIC = "metric"
    PIXEL_NORMALIZATION_RATE = "pixel_normalization_rate"
    TRAINING_SET_LIMIT = "training_set_limit"
    DIMENSIONALITY_REDUCTION_ALGORITHM = "dimensionality_reduction_algorithm"
    DIMENSIONALITY_REDUCTION_N_COMPONENTS = "dimensionality_reduction_n_components"


@dataclass
class FieldConfig:
    """Konfiguracja dla pojedynczego pola w generatorze testów"""
    field_name: KNNTestConfigField
    start: Any
    stop: Any
    step: Any