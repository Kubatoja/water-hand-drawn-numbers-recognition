from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from Testers.Shared.configs import BaseTestConfig, FloodConfig, DimensionalityReductionAlgorithm


@dataclass
class TabICLTestConfig:
    """Konfiguracja dla pojedynczego testu TabICLv2"""

    # Parametry TabICL
    n_estimators: int  # liczba członków ensemblingu (więcej = lepiej, ale wolniej)

    # Parametry wektorów
    training_set_limit: int

    # Informacje o datasecie
    class_count: int

    # Pola z domyślnymi wartościami
    softmax_temperature: float = 0.9   # temperatura dla kontroli pewności predykcji
    outlier_threshold: float = 4.0     # próg z-score dla wykrywania outlierów
    device: str = "auto"               # "auto", "cpu", "cuda"
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
        if not (0.0 < self.softmax_temperature <= 2.0):
            raise ValueError(f"softmax_temperature musi być w (0, 2], otrzymano: {self.softmax_temperature}")
        if self.outlier_threshold <= 0:
            raise ValueError(f"outlier_threshold musi być > 0, otrzymano: {self.outlier_threshold}")
        if self.dimensionality_reduction_algorithm == DimensionalityReductionAlgorithm.FLOOD_FILL and not (0.0 <= self.pixel_normalization_rate <= 1.0):
            raise ValueError(
                f"Pixel normalization rate musi być w [0, 1] dla FLOOD_FILL, otrzymano: {self.pixel_normalization_rate}"
            )
        if self.dimensionality_reduction_n_components <= 0:
            raise ValueError(f"Dimensionality reduction n_components musi być > 0, otrzymano: {self.dimensionality_reduction_n_components}")
        if self.dimensionality_reduction_algorithm == DimensionalityReductionAlgorithm.LDA and self.dimensionality_reduction_n_components >= self.class_count:
            raise ValueError(f"Dla LDA n_components musi być < class_count ({self.class_count}), otrzymano: {self.dimensionality_reduction_n_components}")


class TabICLTestConfigField(Enum):
    """Enum dla pól konfiguracji TabICL"""
    N_ESTIMATORS = "n_estimators"
    SOFTMAX_TEMPERATURE = "softmax_temperature"
    OUTLIER_THRESHOLD = "outlier_threshold"
    DEVICE = "device"
    PIXEL_NORMALIZATION_RATE = "pixel_normalization_rate"
    TRAINING_SET_LIMIT = "training_set_limit"
    DIMENSIONALITY_REDUCTION_ALGORITHM = "dimensionality_reduction_algorithm"
    DIMENSIONALITY_REDUCTION_N_COMPONENTS = "dimensionality_reduction_n_components"


@dataclass
class FieldConfig:
    """Konfiguracja dla pojedynczego pola w generatorze testów"""
    field_name: TabICLTestConfigField
    start: Any
    stop: Any
    step: Any
