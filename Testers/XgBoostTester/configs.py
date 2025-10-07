from dataclasses import dataclass
from enum import Enum
from typing import Any

from Testers.Shared.configs import FloodConfig


@dataclass
class XGBTestConfig:
    """Konfiguracja dla pojedynczego testu XGBoost"""

    # Parametry XGBoost
    learning_rate: float
    n_estimators: int
    max_depth: int
    min_child_weight: float
    gamma: float
    subsample: float
    colsample_bytree: float
    reg_lambda: float  # lambda jest słowem kluczowym w Python
    reg_alpha: float   # alpha -> reg_alpha dla spójności

    # Parametry wektorów
    num_segments: int
    pixel_normalization_rate: float
    training_set_limit: int
    flood_config: FloodConfig

    # Informacje o datasecie
    class_count: int
    image_size: int = 28  # Rozmiar obrazu (domyślnie 28x28)

    def __post_init__(self):
        """Walidacja konfiguracji po inicjalizacji"""
        if self.learning_rate <= 0:
            raise ValueError(f"Learning rate musi być > 0, otrzymano: {self.learning_rate}")
        if self.n_estimators <= 0:
            raise ValueError(f"N estimators musi być > 0, otrzymano: {self.n_estimators}")
        if self.max_depth <= 0:
            raise ValueError(f"Max depth musi być > 0, otrzymano: {self.max_depth}")
        if self.min_child_weight < 0:
            raise ValueError(f"Min child weight musi być >= 0, otrzymano: {self.min_child_weight}")
        if self.gamma < 0:
            raise ValueError(f"Gamma musi być >= 0, otrzymano: {self.gamma}")
        if not (0.0 < self.subsample <= 1.0):
            raise ValueError(f"Subsample musi być w (0, 1], otrzymano: {self.subsample}")
        if not (0.0 < self.colsample_bytree <= 1.0):
            raise ValueError(f"Colsample_bytree musi być w (0, 1], otrzymano: {self.colsample_bytree}")
        if self.reg_lambda < 0:
            raise ValueError(f"Reg lambda musi być >= 0, otrzymano: {self.reg_lambda}")
        if self.reg_alpha < 0:
            raise ValueError(f"Reg alpha musi być >= 0, otrzymano: {self.reg_alpha}")
        if self.num_segments <= 0:
            raise ValueError(f"Num segments musi być > 0, otrzymano: {self.num_segments}")
        if not (0.0 <= self.pixel_normalization_rate <= 1.0):
            raise ValueError(
                f"Pixel normalization rate musi być w [0, 1], otrzymano: {self.pixel_normalization_rate}"
            )


class XGBTestConfigField(Enum):
    """Enum dla pól konfiguracji XGBoost"""
    LEARNING_RATE = "learning_rate"
    N_ESTIMATORS = "n_estimators"
    MAX_DEPTH = "max_depth"
    MIN_CHILD_WEIGHT = "min_child_weight"
    GAMMA = "gamma"
    SUBSAMPLE = "subsample"
    COLSAMPLE_BYTREE = "colsample_bytree"
    REG_LAMBDA = "reg_lambda"
    REG_ALPHA = "reg_alpha"
    NUM_SEGMENTS = "num_segments"
    PIXEL_NORMALIZATION_RATE = "pixel_normalization_rate"
    TRAINING_SET_LIMIT = "training_set_limit"
    FLOOD_CONFIG = "flood_config"


@dataclass
class FieldConfig:
    """Konfiguracja dla pojedynczego pola w generatorze testów"""
    field_name: XGBTestConfigField
    start: Any
    stop: Any
    step: Any
