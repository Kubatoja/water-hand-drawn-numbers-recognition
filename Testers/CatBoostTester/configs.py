from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from Testers.Shared.configs import BaseTestConfig, FloodConfig, DimensionalityReductionAlgorithm


@dataclass
class CatBoostTestConfig:
    """Konfiguracja dla pojedynczego testu CatBoost"""

    learning_rate: float
    n_estimators: int
    max_depth: int
    subsample: float
    colsample_bylevel: float
    random_state: int
    verbose: int
    thread_count: int

    training_set_limit: int
    class_count: int

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
        if self.learning_rate <= 0:
            raise ValueError(f"Learning rate musi być > 0, otrzymano: {self.learning_rate}")
        if self.n_estimators <= 0:
            raise ValueError(f"N estimators musi być > 0, otrzymano: {self.n_estimators}")
        if self.max_depth <= 0:
            raise ValueError(f"Max depth musi być > 0, otrzymano: {self.max_depth}")
        if not (0.0 < self.subsample <= 1.0):
            raise ValueError(f"Subsample musi być w (0, 1], otrzymano: {self.subsample}")
        if not (0.0 < self.colsample_bylevel <= 1.0):
            raise ValueError(f"Colsample_bylevel musi być w (0, 1], otrzymano: {self.colsample_bylevel}")
        if self.thread_count == 0:
            raise ValueError(f"Thread_count nie może być 0, otrzymano: {self.thread_count}")
        if self.dimensionality_reduction_algorithm == DimensionalityReductionAlgorithm.FLOOD_FILL and not (0.0 <= self.pixel_normalization_rate <= 1.0):
            raise ValueError(
                f"Pixel normalization rate musi być w [0, 1] dla FLOOD_FILL, otrzymano: {self.pixel_normalization_rate}"
            )
        if self.dimensionality_reduction_n_components <= 0:
            raise ValueError(f"Dimensionality reduction n_components musi być > 0, otrzymano: {self.dimensionality_reduction_n_components}")
        if self.dimensionality_reduction_algorithm == DimensionalityReductionAlgorithm.LDA and self.dimensionality_reduction_n_components >= self.class_count:
            raise ValueError(f"Dla LDA n_components musi być < class_count ({self.class_count}), otrzymano: {self.dimensionality_reduction_n_components}")


class CatBoostTestConfigField(Enum):
    LEARNING_RATE = "learning_rate"
    N_ESTIMATORS = "n_estimators"
    MAX_DEPTH = "max_depth"
    SUBSAMPLE = "subsample"
    COLSAMPLE_BYLEVEL = "colsample_bylevel"
    RANDOM_STATE = "random_state"
    VERBOSE = "verbose"
    THREAD_COUNT = "thread_count"
    PIXEL_NORMALIZATION_RATE = "pixel_normalization_rate"
    TRAINING_SET_LIMIT = "training_set_limit"
    FLOOD_CONFIG = "flood_config"
    DIMENSIONALITY_REDUCTION_ALGORITHM = "dimensionality_reduction_algorithm"
    DIMENSIONALITY_REDUCTION_N_COMPONENTS = "dimensionality_reduction_n_components"


@dataclass
class FieldConfig:
    field_name: CatBoostTestConfigField
    start: Any
    stop: Any
    step: Any
