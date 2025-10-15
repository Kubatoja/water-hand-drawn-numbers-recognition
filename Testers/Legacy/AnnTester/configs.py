from dataclasses import dataclass
from enum import Enum
from typing import Any

from Testers.Shared.configs import BaseTestConfig, FloodConfig, DimensionalityReductionAlgorithm


@dataclass
class ANNTestConfig:
    """Configuration for a single ANN test case"""

    #ann
    trees_count: int
    leaves_count: int

    #vector
    num_segments: int
    pixel_normalization_rate: float = 0.5  # Domyślnie 0.5, None dla metod statystycznych
    training_set_limit: int
    flood_config: FloodConfig

    # Redukcja wymiarów
    dimensionality_reduction_algorithm: DimensionalityReductionAlgorithm = DimensionalityReductionAlgorithm.NONE
    dimensionality_reduction_n_components: int = 50  # Liczba komponentów do redukcji

    #dataset info
    class_count: int
    image_size: int = 28  # Rozmiar obrazu (domyślnie 28x28)
    dataset_name: str = "Unknown"  # Nazwa datasetu dla raportów

    def __post_init__(self):
        """Walidacja konfiguracji po inicjalizacji"""
        if self.trees_count <= 0:
            raise ValueError(f"Trees count musi być > 0, otrzymano: {self.trees_count}")
        if self.leaves_count <= 0:
            raise ValueError(f"Leaves count musi być > 0, otrzymano: {self.leaves_count}")
        if self.num_segments <= 0:
            raise ValueError(f"Num segments musi być > 0, otrzymano: {self.num_segments}")
        if self.dimensionality_reduction_algorithm == DimensionalityReductionAlgorithm.FLOOD_FILL and not (0.0 <= self.pixel_normalization_rate <= 1.0):
            raise ValueError(
                f"Pixel normalization rate musi być w [0, 1] dla FLOOD_FILL, otrzymano: {self.pixel_normalization_rate}"
            )
        if self.dimensionality_reduction_n_components <= 0:
            raise ValueError(f"Dimensionality reduction n_components musi być > 0, otrzymano: {self.dimensionality_reduction_n_components}")
        if self.dimensionality_reduction_algorithm == DimensionalityReductionAlgorithm.LDA and self.dimensionality_reduction_n_components >= self.class_count:
            raise ValueError(f"Dla LDA n_components musi być < class_count ({self.class_count}), otrzymano: {self.dimensionality_reduction_n_components}")


class ANNTestConfigField(Enum):
    """Enum dla pól konfiguracji ANN"""
    TREES_COUNT = "trees_count"
    LEAVES_COUNT = "leaves_count"
    NUM_SEGMENTS = "num_segments"
    PIXEL_NORMALIZATION_RATE = "pixel_normalization_rate"
    TRAINING_SET_LIMIT = "training_set_limit"
    FLOOD_CONFIG = "flood_config"
    DIMENSIONALITY_REDUCTION_ALGORITHM = "dimensionality_reduction_algorithm"
    DIMENSIONALITY_REDUCTION_N_COMPONENTS = "dimensionality_reduction_n_components"


@dataclass
class FieldConfig:
    """Konfiguracja dla pojedynczego pola w generatorze testów"""
    field_name: ANNTestConfigField
    start: Any
    stop: Any
    step: Any

@dataclass
class FieldConfig:
    field_name: ANNTestConfigField
    start: Any
    stop: Any
    step: Any