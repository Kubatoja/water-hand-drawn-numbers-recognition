from dataclasses import dataclass
from enum import Enum
from typing import Any

from Testers.Shared.configs import FloodConfig


@dataclass
class ANNTestConfig:
    """Configuration for a single ANN test case"""

    #ann
    trees_count: int
    leaves_count: int

    #vector
    num_segments: int
    pixel_normalization_rate: float
    training_set_limit: int
    flood_config: FloodConfig

    #dataset info
    class_count: int

    def __post_init__(self):
        """Walidacja konfiguracji po inicjalizacji"""
        if self.trees_count <= 0:
            raise ValueError(f"Trees count musi być > 0, otrzymano: {self.trees_count}")
        if self.leaves_count <= 0:
            raise ValueError(f"Leaves count musi być > 0, otrzymano: {self.leaves_count}")
        if self.num_segments <= 0:
            raise ValueError(f"Num segments musi być > 0, otrzymano: {self.num_segments}")
        if not (0.0 <= self.pixel_normalization_rate <= 1.0):
            raise ValueError(
                f"Pixel normalization rate musi być w [0, 1], otrzymano: {self.pixel_normalization_rate}"
            )


class ANNTestConfigField(Enum):
    """Enum dla pól konfiguracji ANN"""
    TREES_COUNT = "trees_count"
    LEAVES_COUNT = "leaves_count"
    NUM_SEGMENTS = "num_segments"
    PIXEL_NORMALIZATION_RATE = "pixel_normalization_rate"
    TRAINING_SET_LIMIT = "training_set_limit"
    FLOOD_CONFIG = "flood_config"


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