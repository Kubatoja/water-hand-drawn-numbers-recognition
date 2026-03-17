from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from ..Shared.configs import FloodConfig, DimensionalityReductionAlgorithm


@dataclass
class TabRTestConfig:
    """
    Konfiguracja dla pojedynczego testu TabR.

    Parametry modelu odpowiadają argumentom konstruktora klasy TabR_S_D_Classifier
    z biblioteki pytabkit (pip install pytabkit[models], ICLR 2024).

    Uwaga instalacyjna: TabR wymaga faiss, dostępnego wyłącznie przez conda:
        conda install -c pytorch faiss-cpu   (lub faiss-gpu)
        pip install pytabkit[models]

    Dokumentacja: https://github.com/dholzmueller/pytabkit
    """

    # ------------------------------------------------------------------ #
    #  Podstawowe parametry TabR (pola wymagane)                          #
    # ------------------------------------------------------------------ #
    n_epochs: int
    """Maksymalna liczba epok treningowych."""

    batch_size: int
    """Rozmiar mini-batcha podczas treningu."""

    learning_rate: float
    """Współczynnik uczenia dla optymalizatora Adam (parametr `lr` w pytabkit)."""

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

    device: str = 'auto'
    """
    Urządzenie do trenowania: 'cpu', 'cuda', 'cuda:0' lub 'auto'
    (automatyczny wybór GPU jeśli dostępne).
    """

    verbose: int = 0
    """Poziom szczegółowości logowania (0 = cicho, 1 = verbose)."""

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

    classifier_name: str = "TabR"
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

        if self.n_epochs < 1:
            raise ValueError(
                f"n_epochs musi być >= 1, otrzymano: {self.n_epochs}"
            )
        if self.batch_size < 1:
            raise ValueError(
                f"batch_size musi być >= 1, otrzymano: {self.batch_size}"
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
        valid_devices = ('cpu', 'cuda', 'auto')
        if not (self.device in valid_devices or self.device.startswith('cuda:')):
            raise ValueError(
                f"device musi być jednym z {valid_devices} lub 'cuda:N', "
                f"otrzymano: {self.device}"
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


class TabRTestConfigField(Enum):
    """Enum dla pól konfiguracji TabR – używany przez generator testów."""

    # Podstawowe parametry TabR
    N_EPOCHS = "n_epochs"
    BATCH_SIZE = "batch_size"
    LEARNING_RATE = "learning_rate"

    # Infrastruktura
    RANDOM_STATE = "random_state"
    DEVICE = "device"

    # Preprocessing
    PIXEL_NORMALIZATION_RATE = "pixel_normalization_rate"
    TRAINING_SET_LIMIT = "training_set_limit"
    DIMENSIONALITY_REDUCTION_ALGORITHM = "dimensionality_reduction_algorithm"
    DIMENSIONALITY_REDUCTION_N_COMPONENTS = "dimensionality_reduction_n_components"


@dataclass
class FieldConfig:
    """Konfiguracja zakresu wartości dla pojedynczego pola w generatorze testów."""

    field_name: TabRTestConfigField
    """Pole konfiguracji, które ma być iterowane."""

    start: Any
    """Wartość początkowa zakresu (włącznie)."""

    stop: Any
    """Wartość końcowa zakresu (wyłącznie dla liczb, włącznie dla list)."""

    step: Any
    """Krok iteracji (np. 0.1 dla float, 1 dla int)."""