from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from Testers.Shared.configs import BaseTestConfig, FloodConfig, DimensionalityReductionAlgorithm


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
    training_set_limit: int

    # Informacje o datasecie
    class_count: int

    # Pola z domyślnymi wartościami
    pixel_normalization_rate: float = 0.5  # Domyślnie 0.5, None dla metod statystycznych
    dimensionality_reduction_algorithm: DimensionalityReductionAlgorithm = DimensionalityReductionAlgorithm.NONE
    dimensionality_reduction_n_components: int = 50  # Liczba komponentów do redukcji
    image_size: int = 28  # Rozmiar obrazu (domyślnie 28x28)
    dataset_name: str = "Unknown"  # Nazwa datasetu dla raportów
    classifier_name: str = "Unknown"  # Nazwa klasyfikatora dla raportów
    reduction_name: str = "Unknown"  # Nazwa metody redukcji dla raportów
    num_segments: int = 7  # Domyślna wartość dla Flood Fill
    flood_config: FloodConfig = field(default_factory=lambda: FloodConfig.from_string("1111"))  # Domyślna wartość dla Flood Fill

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