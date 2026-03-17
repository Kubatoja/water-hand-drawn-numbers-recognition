from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from Testers.Shared.configs import FloodConfig, DimensionalityReductionAlgorithm


@dataclass
class GRANDETestConfig:
    """
    Konfiguracja dla pojedynczego testu GRANDE.

    Parametry modelu odpowiadają argumentom konstruktora klasy GRANDE
    z oficjalnej biblioteki (pip install GRANDE, ICLR 2024).

    Dokumentacja parametrów: https://github.com/s-marton/GRANDE
    """

    # ------------------------------------------------------------------ #
    #  Podstawowe parametry GRANDE (pola wymagane)                        #
    # ------------------------------------------------------------------ #
    n_estimators: int
    """Liczba drzew w zbiorze (ensemble size)."""

    max_depth: int
    """Maksymalna głębokość każdego drzewa decyzyjnego."""

    learning_rate: float
    """Współczynnik uczenia dla optymalizatora Adam."""

    # ------------------------------------------------------------------ #
    #  Parametry datasetu (pola wymagane)                                 #
    # ------------------------------------------------------------------ #
    training_set_limit: int
    """Maksymalna liczba próbek treningowych używanych w danym teście."""

    class_count: int
    """Liczba klas klasyfikacji."""

    # ------------------------------------------------------------------ #
    #  Pola z wartościami domyślnymi                                      #
    # ------------------------------------------------------------------ #
    random_state: int = 42
    """Ziarno losowości dla reprodukowalności wyników."""

    verbose: int = 0
    """Poziom szczegółowości logowania GRANDE (0 = cicho, 1 = verbose)."""

    # ------------------------------------------------------------------ #
    #  Parametry preprocessingu i redukcji wymiarów (spójne z SVMConfig) #
    # ------------------------------------------------------------------ #
    pixel_normalization_rate: float = 0.5
    """Stopień normalizacji pikseli (0.0 – 1.0). Używane przy Flood Fill."""

    dimensionality_reduction_algorithm: DimensionalityReductionAlgorithm = DimensionalityReductionAlgorithm.NONE
    """Algorytm redukcji wymiarów stosowany przed klasyfikatorem."""

    dimensionality_reduction_n_components: int = 50
    """Liczba komponentów dla algorytmu redukcji wymiarów."""

    image_size: int = 28
    """Rozmiar obrazu wejściowego (domyślnie 28×28 dla MNIST)."""

    dataset_name: str = "Unknown"
    """Nazwa datasetu – używana w raportach i logach."""

    classifier_name: str = "GRANDE"
    """Nazwa klasyfikatora – używana w raportach i logach."""

    reduction_name: str = "Unknown"
    """Nazwa metody redukcji wymiarów – używana w raportach i logach."""

    num_segments: int = 7
    """Liczba segmentów dla algorytmu Flood Fill."""

    flood_config: FloodConfig = field(
        default_factory=lambda: FloodConfig.from_string("1111")
    )
    """Konfiguracja Flood Fill."""

    def __post_init__(self) -> None:
        """Walidacja konfiguracji po inicjalizacji."""

        if self.n_estimators < 1:
            raise ValueError(
                f"n_estimators musi być >= 1, otrzymano: {self.n_estimators}"
            )
        if self.max_depth < 1:
            raise ValueError(
                f"max_depth musi być >= 1, otrzymano: {self.max_depth}"
            )
        if not (0.0 < self.learning_rate <= 1.0):
            raise ValueError(
                f"learning_rate musi być w przedziale (0, 1], otrzymano: {self.learning_rate}"
            )
        if self.training_set_limit < 1:
            raise ValueError(
                f"training_set_limit musi być >= 1, otrzymano: {self.training_set_limit}"
            )
        if self.class_count < 2:
            raise ValueError(
                f"class_count musi być >= 2, otrzymano: {self.class_count}"
            )
        if (
            self.dimensionality_reduction_algorithm == DimensionalityReductionAlgorithm.FLOOD_FILL
            and not (0.0 <= self.pixel_normalization_rate <= 1.0)
        ):
            raise ValueError(
                f"pixel_normalization_rate musi być w [0, 1] dla FLOOD_FILL, "
                f"otrzymano: {self.pixel_normalization_rate}"
            )
        if self.dimensionality_reduction_n_components <= 0:
            raise ValueError(
                f"dimensionality_reduction_n_components musi być > 0, "
                f"otrzymano: {self.dimensionality_reduction_n_components}"
            )
        if (
            self.dimensionality_reduction_algorithm == DimensionalityReductionAlgorithm.LDA
            and self.dimensionality_reduction_n_components >= self.class_count
        ):
            raise ValueError(
                f"Dla LDA n_components musi być < class_count ({self.class_count}), "
                f"otrzymano: {self.dimensionality_reduction_n_components}"
            )


class GRANDETestConfigField(Enum):
    """Enum dla pól konfiguracji GRANDE – używany przez generator testów."""

    # Podstawowe parametry GRANDE
    N_ESTIMATORS = "n_estimators"
    MAX_DEPTH = "max_depth"
    LEARNING_RATE = "learning_rate"

    # Trening
    RANDOM_STATE = "random_state"

    # Preprocessing
    PIXEL_NORMALIZATION_RATE = "pixel_normalization_rate"
    TRAINING_SET_LIMIT = "training_set_limit"
    DIMENSIONALITY_REDUCTION_ALGORITHM = "dimensionality_reduction_algorithm"
    DIMENSIONALITY_REDUCTION_N_COMPONENTS = "dimensionality_reduction_n_components"


@dataclass
class FieldConfig:
    """Konfiguracja zakresu wartości dla pojedynczego pola w generatorze testów."""

    field_name: GRANDETestConfigField
    """Pole konfiguracji, które ma być iterowane."""

    start: Any
    """Wartość początkowa zakresu (włącznie)."""

    stop: Any
    """Wartość końcowa zakresu (wyłącznie dla liczb, włącznie dla list)."""

    step: Any
    """Krok iteracji (np. 0.1 dla float, 1 dla int)."""