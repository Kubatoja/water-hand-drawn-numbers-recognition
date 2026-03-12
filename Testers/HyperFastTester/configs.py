from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from Testers.Shared.configs import BaseTestConfig, FloodConfig, DimensionalityReductionAlgorithm


@dataclass
class HyperFastTestConfig:
    """Konfiguracja dla pojedynczego testu HyperFast"""

    # Parametry wektorów
    training_set_limit: int

    # Informacje o datasecie
    class_count: int

    # Parametry HyperFast
    n_ensemble: int = 16            # liczba członków ensemblingu
    batch_size: int = 2048          # rozmiar batcha przy inferencji (0 = wszystko naraz)
    nn_bias: float = 0.0            # bias dla nearest-neighbor computation
    optimization: str = "optimize"  # "optimize" lub "fit_only"
    optimize_steps: int = 64        # liczba kroków optymalizacji

    # Pola wspólne
    device: str = "auto"            # "auto", "cpu", "cuda"
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
        if self.n_ensemble <= 0:
            raise ValueError(f"n_ensemble musi być > 0, otrzymano: {self.n_ensemble}")
        if self.batch_size < 0:
            raise ValueError(f"batch_size musi być >= 0, otrzymano: {self.batch_size}")
        if self.optimize_steps < 0:
            raise ValueError(f"optimize_steps musi być >= 0, otrzymano: {self.optimize_steps}")
        if self.optimization not in ("optimize", "fit_only"):
            raise ValueError(f"optimization musi być 'optimize' lub 'fit_only', otrzymano: {self.optimization}")
        if self.dimensionality_reduction_algorithm == DimensionalityReductionAlgorithm.FLOOD_FILL and not (0.0 <= self.pixel_normalization_rate <= 1.0):
            raise ValueError(
                f"Pixel normalization rate musi być w [0, 1] dla FLOOD_FILL, otrzymano: {self.pixel_normalization_rate}"
            )
        if self.dimensionality_reduction_n_components <= 0:
            raise ValueError(f"Dimensionality reduction n_components musi być > 0, otrzymano: {self.dimensionality_reduction_n_components}")
        if self.dimensionality_reduction_algorithm == DimensionalityReductionAlgorithm.LDA and self.dimensionality_reduction_n_components >= self.class_count:
            raise ValueError(f"Dla LDA n_components musi być < class_count ({self.class_count}), otrzymano: {self.dimensionality_reduction_n_components}")


class HyperFastTestConfigField(Enum):
    """Enum dla pól konfiguracji HyperFast"""
    N_ENSEMBLE = "n_ensemble"
    BATCH_SIZE = "batch_size"
    NN_BIAS = "nn_bias"
    OPTIMIZATION = "optimization"
    OPTIMIZE_STEPS = "optimize_steps"
    DEVICE = "device"
    PIXEL_NORMALIZATION_RATE = "pixel_normalization_rate"
    TRAINING_SET_LIMIT = "training_set_limit"
    DIMENSIONALITY_REDUCTION_ALGORITHM = "dimensionality_reduction_algorithm"
    DIMENSIONALITY_REDUCTION_N_COMPONENTS = "dimensionality_reduction_n_components"


@dataclass
class FieldConfig:
    """Konfiguracja dla pojedynczego pola w generatorze testów"""
    field_name: HyperFastTestConfigField
    start: Any
    stop: Any
    step: Any
