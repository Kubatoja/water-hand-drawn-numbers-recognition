from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from Testers.Shared.configs import BaseTestConfig, FloodConfig, DimensionalityReductionAlgorithm


@dataclass
class TabPFNTestConfig:
    """Konfiguracja dla pojedynczego testu TabPFN-v2"""

    # Parametry TabPFN
    n_estimators: int  # liczba estymatorów w ensemblingu
    use_v2: bool       # True = wagi TabPFN-v2 (Apache 2.0), False = domyślne wagi

    # Parametry wektorów
    training_set_limit: int

    # Informacje o datasecie
    class_count: int

    # Pola z domyślnymi wartościami
    device: str = "auto"  # "auto", "cpu", "cuda"
    random_state: int = 42
    pixel_normalization_rate: float = 0.5
    dimensionality_reduction_algorithm: DimensionalityReductionAlgorithm = DimensionalityReductionAlgorithm.NONE
    dimensionality_reduction_n_components: int = 50
    image_size: int = 28
    dataset_name: str = "Unknown"
    classifier_name: str = "Unknown"
    reduction_name: str = "Unknown"
    num_segments: int = 7
    flood_config: FloodConfig = field(default_factory=lambda: FloodConfig.from_string("1111"))

    def __post_init__(self):
        """Walidacja konfiguracji po inicjalizacji"""
        if self.n_estimators <= 0:
            raise ValueError(f"n_estimators musi być > 0, otrzymano: {self.n_estimators}")
        if self.dimensionality_reduction_algorithm == DimensionalityReductionAlgorithm.FLOOD_FILL and not (0.0 <= self.pixel_normalization_rate <= 1.0):
            raise ValueError(
                f"Pixel normalization rate musi być w [0, 1] dla FLOOD_FILL, otrzymano: {self.pixel_normalization_rate}"
            )
        if self.dimensionality_reduction_n_components <= 0:
            raise ValueError(f"Dimensionality reduction n_components musi być > 0, otrzymano: {self.dimensionality_reduction_n_components}")
        if self.dimensionality_reduction_algorithm == DimensionalityReductionAlgorithm.LDA and self.dimensionality_reduction_n_components >= self.class_count:
            raise ValueError(f"Dla LDA n_components musi być < class_count ({self.class_count}), otrzymano: {self.dimensionality_reduction_n_components}")


class TabPFNTestConfigField(Enum):
    """Enum dla pól konfiguracji TabPFN"""
    N_ESTIMATORS = "n_estimators"
    USE_V2 = "use_v2"
    DEVICE = "device"
    PIXEL_NORMALIZATION_RATE = "pixel_normalization_rate"
    TRAINING_SET_LIMIT = "training_set_limit"
    DIMENSIONALITY_REDUCTION_ALGORITHM = "dimensionality_reduction_algorithm"
    DIMENSIONALITY_REDUCTION_N_COMPONENTS = "dimensionality_reduction_n_components"


@dataclass
class FieldConfig:
    """Konfiguracja dla pojedynczego pola w generatorze testów"""
    field_name: TabPFNTestConfigField
    start: Any
    stop: Any
    step: Any
