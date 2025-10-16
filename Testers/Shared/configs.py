from dataclasses import dataclass
from enum import Enum


@dataclass
class TestRunnerConfig:
    """Konfiguracja dla TestRunner - uniwersalna dla wszystkich algorytmów"""
    skip_first_vector_generation: bool = False
    save_results_after_each_test: bool = False


class FloodSide(Enum):
    """Enum dla kierunków flood fill"""
    LEFT = 0
    RIGHT = 1
    TOP = 2
    BOTTOM = 3


class DimensionalityReductionAlgorithm(Enum):
    """Enum dla algorytmów redukcji wymiarów"""
    NONE = "none"
    FLOOD_FILL = "flood_fill"
    PCA = "pca"
    LDA = "lda"
    ISOMAP = "isomap"
    TSNE = "tsne"


@dataclass
class BaseTestConfig:
    """Bazowa konfiguracja testu wspólna dla wszystkich algorytmów"""
    
    # Informacje o datasecie (wymagane)
    class_count: int
    
    # Parametry wektorów
    pixel_normalization_rate: float = 0.5  # Domyślnie 0.5, None dla metod statystycznych
    training_set_limit: int = 1000  # Domyślny limit zbioru treningowego
    
    # Redukcja wymiarów
    dimensionality_reduction_algorithm: DimensionalityReductionAlgorithm = DimensionalityReductionAlgorithm.NONE
    dimensionality_reduction_n_components: int = 50  # Liczba komponentów do redukcji

    # Informacje o datasecie
    image_size: int = 28  # Rozmiar obrazu (domyślnie 28x28)
    dataset_name: str = "Unknown"  # Nazwa datasetu dla raportów


@dataclass
class ReductionConfig:
    """Konfiguracja redukcji wymiarów"""
    name: str
    algorithm: DimensionalityReductionAlgorithm
    n_components: int
    training_set_limit: int
    requires_bfs: bool = False
    num_segments: int = 7
    pixel_normalization_rate: float = 0.5
    flood_config: 'FloodConfig' = None

    def __post_init__(self):
        if self.flood_config is None:
            self.flood_config = FloodConfig()


@dataclass
class FloodConfig:
    """Konfiguracja dla flood fill - używana w generowaniu wektorów"""
    left: bool = True
    right: bool = True
    top: bool = True
    bottom: bool = True

    @classmethod
    def from_string(cls, flood_string: str) -> 'FloodConfig':
        """Tworzy FloodConfig ze stringa typu '1111'"""
        if len(flood_string) != 4:
            raise ValueError(
                f"Flood string musi mieć 4 znaki, otrzymano: {flood_string}"
            )

        return cls(
            left=flood_string[0] == '1',
            right=flood_string[1] == '1',
            top=flood_string[2] == '1',
            bottom=flood_string[3] == '1'
        )

    def to_string(self) -> str:
        """Konwertuje do formatu string '1111'"""
        return (
            ('1' if self.left else '0') +
            ('1' if self.right else '0') +
            ('1' if self.top else '0') +
            ('1' if self.bottom else '0')
        )

    def to_human_readable(self) -> str:
        """Konwertuje do formatu czytelnego dla człowieka"""
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
