from dataclasses import dataclass
from enum import Enum
from typing import Any


@dataclass
class TestRunnerConfig:
    """Konfiguracja dla TestRunner"""
    skip_first_vector_generation: bool = False
    save_results_after_each_test: bool = False

class FloodSide(Enum):
    """Enum for flood side directions"""
    LEFT = 0
    RIGHT = 1
    TOP = 2
    BOTTOM = 3


@dataclass
class FloodConfig:
    """Configuration for flood fill sides"""
    left: bool = True
    right: bool = True
    top: bool = True
    bottom: bool = True

    @classmethod
    def from_string(cls, flood_string: str) -> 'FloodConfig':
        """Create FloodConfig from string like '1111'"""
        if len(flood_string) != 4:
            raise ValueError(f"Flood string must be 4 characters, got: {flood_string}")

        return cls(
            left=flood_string[0] == '1',
            right=flood_string[1] == '1',
            top=flood_string[2] == '1',
            bottom=flood_string[3] == '1'
        )

    def to_string(self) -> str:
        """Convert to string format like '1111'"""
        return (
                ('1' if self.left else '0') +
                ('1' if self.right else '0') +
                ('1' if self.top else '0') +
                ('1' if self.bottom else '0')
        )

    def to_human_readable(self) -> str:
        """Convert to human readable format"""
        sides = []
        if self.left:
            sides.append("left")
        if self.right:
            sides.append("right")
        if self.top:
            sides.append("top")
        if self.bottom:
            sides.append("bottom")

        return ", ".join(sides) + f", {self.to_string()}"
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
        """Validate configuration after initialization"""
        if self.trees_count <= 0:
            raise ValueError(f"Trees count must be positive, got: {self.trees_count}")
        if self.leaves_count <= 0:
            raise ValueError(f"Leaves count must be positive, got: {self.leaves_count}")
        if self.num_segments <= 0:
            raise ValueError(f"Number of segments must be positive, got: {self.num_segments}")
        if not (0.0 <= self.pixel_normalization_rate <= 1.0):
            raise ValueError(f"Pixel normalization rate must be between 0.0-1.0, got: {self.pixel_normalization_rate}")


# Konfiguracja dla pojedynczego pola

class ANNTestConfigField(Enum):
    TREES_COUNT = "trees_count"
    LEAVES_COUNT = "leaves_count"
    NUM_SEGMENTS = "num_segments"
    PIXEL_NORMALIZATION_RATE = "pixel_normalization_rate"
    TRAINING_SET_LIMIT = "training_set_limit"
    FLOOD_CONFIG = "flood_config"

@dataclass
class FieldConfig:
    field_name: ANNTestConfigField
    start: Any
    stop: Any
    step: Any