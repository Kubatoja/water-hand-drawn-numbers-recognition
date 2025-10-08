"""
Enums dla datasetów i ich wariantów.
Odpowiedzialność: Definicja wszystkich dostępnych datasetów w projekcie.
"""
from enum import Enum


class DatasetName(Enum):
    """Główne datasety dostępne w projekcie"""
    MNIST = "mnist"
    FASHION_MNIST = "fashion_mnist"
    EMNIST_BALANCED = "emnist_balanced"
    EMNIST_DIGITS = "emnist_digits"
    ARABIC = "arabic"
    USPS = "usps"
    MNIST_C = "mnist_c"  # Bazowy dla wszystkich wariantów MNIST-C
    
    def __str__(self) -> str:
        return self.value


class MnistCCorruption(Enum):
    """
    Wszystkie warianty corrupted MNIST-C.
    Każdy wariant reprezentuje inny typ zakłócenia/transformacji obrazu.
    """
    IDENTITY = "identity"              # Oryginalne obrazy bez zakłóceń
    BRIGHTNESS = "brightness"          # Zmieniona jasność
    CANNY_EDGES = "canny_edges"       # Detekcja krawędzi Canny
    DOTTED_LINE = "dotted_line"       # Kropkowane linie
    FOG = "fog"                       # Efekt mgły
    GLASS_BLUR = "glass_blur"         # Rozmycie szkła
    IMPULSE_NOISE = "impulse_noise"   # Szum impulsowy
    MOTION_BLUR = "motion_blur"       # Rozmycie ruchu
    ROTATE = "rotate"                 # Rotacja
    SCALE = "scale"                   # Skalowanie
    SHEAR = "shear"                   # Ścinanie
    SHOT_NOISE = "shot_noise"         # Szum Poissona
    SPATTER = "spatter"               # Plamy
    STRIPE = "stripe"                 # Paski
    TRANSLATE = "translate"           # Przesunięcie
    ZIGZAG = "zigzag"                 # Zygzak
    
    def __str__(self) -> str:
        return self.value
    
    @property
    def display_name(self) -> str:
        """Czytelna nazwa dla wyświetlania"""
        return self.value.replace('_', ' ').title()
    
    @classmethod
    def all_corruptions(cls) -> list['MnistCCorruption']:
        """Zwraca wszystkie warianty corrupted (bez identity)"""
        return [c for c in cls if c != cls.IDENTITY]
    
    @classmethod
    def all_including_identity(cls) -> list['MnistCCorruption']:
        """Zwraca wszystkie warianty włącznie z identity"""
        return list(cls)


# Pomocnicze grupy datasetów
class DatasetGroup(Enum):
    """Predefiniowane grupy datasetów do testowania"""
    STANDARD_DIGITS = "standard_digits"      # MNIST, EMNIST-digits, USPS
    ALL_EMNIST = "all_emnist"               # EMNIST-balanced, EMNIST-digits
    ALL_BASIC = "all_basic"                 # Wszystkie podstawowe (bez MNIST-C)
    MNIST_C_ALL = "mnist_c_all"             # Wszystkie warianty MNIST-C
    EVERYTHING = "everything"                # Absolutnie wszystko
    
    def __str__(self) -> str:
        return self.value
