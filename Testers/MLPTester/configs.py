from dataclasses import dataclass
from enum import Enum
from typing import Any, Tuple

from Testers.Shared.configs import BaseTestConfig, FloodConfig, DimensionalityReductionAlgorithm


@dataclass
class MLPTestConfig:
    """Konfiguracja dla pojedynczego testu MLP"""

    # Parametry MLP
    hidden_layer_sizes: Tuple[int, ...]
    activation: str  # 'identity', 'logistic', 'tanh', 'relu'
    solver: str  # 'lbfgs', 'sgd', 'adam'
    alpha: float
    learning_rate: str  # 'constant', 'invscaling', 'adaptive'
    learning_rate_init: float
    max_iter: int

    # Parametry wektorów
    num_segments: int
    training_set_limit: int
    flood_config: FloodConfig

    # Informacje o datasecie
    class_count: int

    # Pola z domyślnymi wartościami
    random_state: int = 42
    pixel_normalization_rate: float = 0.5  # Domyślnie 0.5, None dla metod statystycznych
    dimensionality_reduction_algorithm: DimensionalityReductionAlgorithm = DimensionalityReductionAlgorithm.NONE
    dimensionality_reduction_n_components: int = 50  # Liczba komponentów do redukcji
    image_size: int = 28  # Rozmiar obrazu (domyślnie 28x28)
    dataset_name: str = "Unknown"  # Nazwa datasetu dla raportów

    def __post_init__(self):
        """Walidacja konfiguracji po inicjalizacji"""
        if not self.hidden_layer_sizes or any(size <= 0 for size in self.hidden_layer_sizes):
            raise ValueError(f"Hidden layer sizes muszą być dodatnie, otrzymano: {self.hidden_layer_sizes}")
        if self.activation not in ['identity', 'logistic', 'tanh', 'relu']:
            raise ValueError(f"Activation musi być jednym z: 'identity', 'logistic', 'tanh', 'relu', otrzymano: {self.activation}")
        if self.solver not in ['lbfgs', 'sgd', 'adam']:
            raise ValueError(f"Solver musi być jednym z: 'lbfgs', 'sgd', 'adam', otrzymano: {self.solver}")
        if self.alpha < 0:
            raise ValueError(f"Alpha musi być >= 0, otrzymano: {self.alpha}")
        if self.learning_rate not in ['constant', 'invscaling', 'adaptive']:
            raise ValueError(f"Learning rate musi być jednym z: 'constant', 'invscaling', 'adaptive', otrzymano: {self.learning_rate}")
        if self.learning_rate_init <= 0:
            raise ValueError(f"Learning rate init musi być > 0, otrzymano: {self.learning_rate_init}")
        if self.max_iter <= 0:
            raise ValueError(f"Max iter musi być > 0, otrzymano: {self.max_iter}")
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


class MLPTestConfigField(Enum):
    """Enum dla pól konfiguracji MLP"""
    HIDDEN_LAYER_SIZES = "hidden_layer_sizes"
    ACTIVATION = "activation"
    SOLVER = "solver"
    ALPHA = "alpha"
    LEARNING_RATE = "learning_rate"
    LEARNING_RATE_INIT = "learning_rate_init"
    MAX_ITER = "max_iter"
    RANDOM_STATE = "random_state"
    NUM_SEGMENTS = "num_segments"
    PIXEL_NORMALIZATION_RATE = "pixel_normalization_rate"
    TRAINING_SET_LIMIT = "training_set_limit"
    FLOOD_CONFIG = "flood_config"
    DIMENSIONALITY_REDUCTION_ALGORITHM = "dimensionality_reduction_algorithm"
    DIMENSIONALITY_REDUCTION_N_COMPONENTS = "dimensionality_reduction_n_components"


@dataclass
class FieldConfig:
    """Konfiguracja dla pojedynczego pola w generatorze testów"""
    field_name: MLPTestConfigField
    start: Any
    stop: Any
    step: Any