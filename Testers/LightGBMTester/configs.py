from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from Testers.Shared.configs import BaseTestConfig, FloodConfig, DimensionalityReductionAlgorithm


@dataclass
class LightGBMTestConfig:
    """Konfiguracja dla pojedynczego testu LightGBM"""

    learning_rate: float
    n_estimators: int
    max_depth: int
    num_leaves: int
    subsample: float
    colsample_bytree: float
    random_state: int
    n_jobs: int

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
        if self.num_leaves <= 0:
            raise ValueError(f"Num leaves musi być > 0, otrzymano: {self.num_leaves}")
        if not (0.0 < self.subsample <= 1.0):
            raise ValueError(f"Subsample musi być w (0, 1], otrzymano: {self.subsample}")
        if not (0.0 < self.colsample_bytree <= 1.0):
            raise ValueError(f"Colsample_bytree musi być w (0, 1], otrzymano: {self.colsample_bytree}")
        if self.n_jobs == 0:
            raise ValueError(f"n_jobs nie może być 0, otrzymano: {self.n_jobs}")
        if self.dimensionality_reduction_algorithm == DimensionalityReductionAlgorithm.FLOOD_FILL and not (0.0 <= self.pixel_normalization_rate <= 1.0):
            raise ValueError(
                f"Pixel normalization rate musi być w [0, 1] dla FLOOD_FILL, otrzymano: {self.pixel_normalization_rate}"
            )
        if self.dimensionality_reduction_n_components <= 0:
            raise ValueError(f"Dimensionality reduction n_components musi być > 0, otrzymano: {self.dimensionality_reduction_n_components}")
        if self.dimensionality_reduction_algorithm == DimensionalityReductionAlgorithm.LDA and self.dimensionality_reduction_n_components >= self.class_count:
            raise ValueError(f"Dla LDA n_components musi być < class_count ({self.class_count}), otrzymano: {self.dimensionality_reduction_n_components}")


class LightGBMTestConfigField(Enum):
    LEARNING_RATE = "learning_rate"
    N_ESTIMATORS = "n_estimators"
    MAX_DEPTH = "max_depth"
    NUM_LEAVES = "num_leaves"
    SUBSAMPLE = "subsample"
    COLSAMPLE_BYTREE = "colsample_bytree"
    RANDOM_STATE = "random_state"
    N_JOBS = "n_jobs"
    PIXEL_NORMALIZATION_RATE = "pixel_normalization_rate"
    TRAINING_SET_LIMIT = "training_set_limit"
    FLOOD_CONFIG = "flood_config"
    DIMENSIONALITY_REDUCTION_ALGORITHM = "dimensionality_reduction_algorithm"
    DIMENSIONALITY_REDUCTION_N_COMPONENTS = "dimensionality_reduction_n_components"


@dataclass
class FieldConfig:
    field_name: LightGBMTestConfigField
    start: Any
    stop: Any
    step: Any
