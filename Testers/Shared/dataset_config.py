"""
Dataset Configuration dla Bayesian Optimization.
Odpowiedzialność: Konfiguracja datasetów do testowania.
"""
from dataclasses import dataclass
from typing import List, Optional, Dict
from pathlib import Path

from Testers.Shared.DataLoader import DataType
from Testers.Shared.dataset_enums import DatasetName, MnistCCorruption, DatasetGroup


@dataclass
class DatasetConfig:
    """Konfiguracja pojedynczego datasetu."""
    
    name: str
    train_path: str
    test_path: str
    data_type: DataType
    class_count: int
    image_size: int = 28  # Rozmiar obrazu (np. 28 dla 28x28, 16 dla 16x16)
    train_labels_path: Optional[str] = None  # Dla formatów z oddzielnymi etykietami
    test_labels_path: Optional[str] = None   # Dla formatów z oddzielnymi etykietami
    dataset_enum: Optional[DatasetName] = None  # Powiązanie z enumem
    corruption_type: Optional[MnistCCorruption] = None  # Dla MNIST-C
    
    def __str__(self) -> str:
        if self.corruption_type:
            return f"{self.name} - {self.corruption_type.display_name} ({self.class_count} classes, {self.image_size}x{self.image_size})"
        return f"{self.name} ({self.class_count} classes, {self.image_size}x{self.image_size})"
    
    @property
    def image_shape(self) -> tuple:
        """Zwraca kształt obrazu jako tuple (height, width)"""
        return (self.image_size, self.image_size)
    
    @property
    def display_name(self) -> str:
        """Czytelna nazwa do wyświetlania"""
        if self.corruption_type:
            return f"{self.name} ({self.corruption_type.display_name})"
        return self.name



# ============================================================================
# PREDEFINIOWANE DATASETY
# ============================================================================

# ----- MNIST -----
MNIST_DATASET = DatasetConfig(
    name='MNIST',
    train_path='Data/Mnist/mnist_train.csv',
    test_path='Data/Mnist/mnist_test.csv',
    data_type=DataType.MNIST_FORMAT,
    class_count=10,
    image_size=28,
    dataset_enum=DatasetName.MNIST
)

# ----- Fashion MNIST -----
FASHION_MNIST_DATASET = DatasetConfig(
    name='Fashion-MNIST',
    train_path='Data/FashionMnist/fashion_mnist_train.csv',
    test_path='Data/FashionMnist/fashion_mnist_test.csv',
    data_type=DataType.MNIST_FORMAT,
    class_count=10,
    image_size=28,
    dataset_enum=DatasetName.FASHION_MNIST
)

# ----- EMNIST Balanced -----
EMNIST_BALANCED_DATASET = DatasetConfig(
    name='EMNIST-Balanced',
    train_path='Data/Emnist-balanced/emnist-balanced-train.csv',
    test_path='Data/Emnist-balanced/emnist-balanced-test.csv',
    data_type=DataType.MNIST_FORMAT,
    class_count=47,
    image_size=28,
    dataset_enum=DatasetName.EMNIST_BALANCED
)

# ----- EMNIST Digits -----
EMNIST_DIGITS_DATASET = DatasetConfig(
    name='EMNIST-Digits',
    train_path='Data/Emnist-digits/emnist-digits-train.csv',
    test_path='Data/Emnist-digits/emnist-digits-test.csv',
    data_type=DataType.MNIST_FORMAT,
    class_count=10,
    image_size=28,
    dataset_enum=DatasetName.EMNIST_DIGITS
)

# ----- Arabic Handwritten Digits -----
ARABIC_DATASET = DatasetConfig(
    name='Arabic',
    train_path='Data/Arabic/csvTrainImages 60k x 784.csv',
    test_path='Data/Arabic/csvTestImages 10k x 784.csv',
    train_labels_path='Data/Arabic/csvTrainLabel 60k x 1.csv',
    test_labels_path='Data/Arabic/csvTestLabel 10k x 1.csv',
    data_type=DataType.SEPARATED_FORMAT,
    class_count=10,
    image_size=28,
    dataset_enum=DatasetName.ARABIC
)

# ----- USPS -----
USPS_DATASET = DatasetConfig(
    name='USPS',
    train_path='Data/Usps/usps.bz2',
    test_path='Data/Usps/usps.t.bz2',
    data_type=DataType.LIBSVM_FORMAT,
    class_count=10,
    image_size=16,  # USPS ma 16x16 obrazy
    dataset_enum=DatasetName.USPS
)


# ============================================================================
# MNIST-C (Corrupted MNIST) - Wszystkie warianty
# ============================================================================

def create_mnist_c_config(corruption: MnistCCorruption) -> DatasetConfig:
    """
    Tworzy konfigurację dla konkretnego wariantu MNIST-C.
    
    Args:
        corruption: Typ korupcji z MnistCCorruption enum
    
    Returns:
        DatasetConfig dla danego wariantu MNIST-C
    """
    base_path = f'Data/Mnist-C/mnist_c/{corruption.value}'
    
    return DatasetConfig(
        name=f'MNIST-C',
        train_path=base_path,  # Folder, nie plik
        test_path=base_path,   # Ten sam folder dla train i test
        data_type=DataType.NPY_FORMAT,
        class_count=10,
        image_size=28,
        dataset_enum=DatasetName.MNIST_C,
        corruption_type=corruption
    )


# Wszystkie warianty MNIST-C
MNIST_C_IDENTITY = create_mnist_c_config(MnistCCorruption.IDENTITY)
MNIST_C_BRIGHTNESS = create_mnist_c_config(MnistCCorruption.BRIGHTNESS)
MNIST_C_CANNY_EDGES = create_mnist_c_config(MnistCCorruption.CANNY_EDGES)
MNIST_C_DOTTED_LINE = create_mnist_c_config(MnistCCorruption.DOTTED_LINE)
MNIST_C_FOG = create_mnist_c_config(MnistCCorruption.FOG)
MNIST_C_GLASS_BLUR = create_mnist_c_config(MnistCCorruption.GLASS_BLUR)
MNIST_C_IMPULSE_NOISE = create_mnist_c_config(MnistCCorruption.IMPULSE_NOISE)
MNIST_C_MOTION_BLUR = create_mnist_c_config(MnistCCorruption.MOTION_BLUR)
MNIST_C_ROTATE = create_mnist_c_config(MnistCCorruption.ROTATE)
MNIST_C_SCALE = create_mnist_c_config(MnistCCorruption.SCALE)
MNIST_C_SHEAR = create_mnist_c_config(MnistCCorruption.SHEAR)
MNIST_C_SHOT_NOISE = create_mnist_c_config(MnistCCorruption.SHOT_NOISE)
MNIST_C_SPATTER = create_mnist_c_config(MnistCCorruption.SPATTER)
MNIST_C_STRIPE = create_mnist_c_config(MnistCCorruption.STRIPE)
MNIST_C_TRANSLATE = create_mnist_c_config(MnistCCorruption.TRANSLATE)
MNIST_C_ZIGZAG = create_mnist_c_config(MnistCCorruption.ZIGZAG)

# Lista wszystkich wariantów MNIST-C
ALL_MNIST_C_DATASETS = [
    MNIST_C_IDENTITY,
    MNIST_C_BRIGHTNESS,
    MNIST_C_CANNY_EDGES,
    MNIST_C_DOTTED_LINE,
    MNIST_C_FOG,
    MNIST_C_GLASS_BLUR,
    MNIST_C_IMPULSE_NOISE,
    MNIST_C_MOTION_BLUR,
    MNIST_C_ROTATE,
    MNIST_C_SCALE,
    MNIST_C_SHEAR,
    MNIST_C_SHOT_NOISE,
    MNIST_C_SPATTER,
    MNIST_C_STRIPE,
    MNIST_C_TRANSLATE,
    MNIST_C_ZIGZAG,
]


# ============================================================================
# KOLEKCJE I GRUPY DATASETÓW
# ============================================================================

# Podstawowe datasety (bez MNIST-C)
BASIC_DATASETS = [
    MNIST_DATASET,
    FASHION_MNIST_DATASET,
    EMNIST_BALANCED_DATASET,
    EMNIST_DIGITS_DATASET,
    ARABIC_DATASET,
    USPS_DATASET,
]

# Tylko datasety z cyframi (10 klas)
DIGITS_ONLY_DATASETS = [
    MNIST_DATASET,
    EMNIST_DIGITS_DATASET,
    ARABIC_DATASET,
    USPS_DATASET,
]

# Tylko datasety z innymi obiektami (10 klas, nie cyfry)
NON_DIGITS_DATASETS = [
    FASHION_MNIST_DATASET,
]

# Wszystkie EMNIST
ALL_EMNIST_DATASETS = [
    EMNIST_BALANCED_DATASET,
    EMNIST_DIGITS_DATASET,
]

# Absolutnie wszystkie datasety
ALL_DATASETS = BASIC_DATASETS + ALL_MNIST_C_DATASETS

# Mapowanie enum -> config dla łatwego dostępu
DATASET_REGISTRY: Dict[DatasetName, List[DatasetConfig]] = {
    DatasetName.MNIST: [MNIST_DATASET],
    DatasetName.FASHION_MNIST: [FASHION_MNIST_DATASET],
    DatasetName.EMNIST_BALANCED: [EMNIST_BALANCED_DATASET],
    DatasetName.EMNIST_DIGITS: [EMNIST_DIGITS_DATASET],
    DatasetName.ARABIC: [ARABIC_DATASET],
    DatasetName.USPS: [USPS_DATASET],
    DatasetName.MNIST_C: ALL_MNIST_C_DATASETS,
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_dataset(dataset_name: DatasetName, 
                corruption: Optional[MnistCCorruption] = None) -> DatasetConfig:
    """
    Pobiera konfigurację datasetu na podstawie enum.
    
    Args:
        dataset_name: Nazwa datasetu z enum
        corruption: Opcjonalnie typ korupcji dla MNIST-C
    
    Returns:
        DatasetConfig dla danego datasetu
    
    Raises:
        ValueError: Jeśli dataset nie został znaleziony
    """
    if dataset_name == DatasetName.MNIST_C:
        if corruption is None:
            raise ValueError("MNIST-C wymaga określenia typu korupcji")
        return create_mnist_c_config(corruption)
    
    configs = DATASET_REGISTRY.get(dataset_name)
    if not configs:
        raise ValueError(f"Dataset {dataset_name} nie został znaleziony")
    
    return configs[0]


def get_datasets_by_group(group: DatasetGroup) -> List[DatasetConfig]:
    """
    Pobiera listę datasetów na podstawie predefiniowanej grupy.
    
    Args:
        group: Grupa datasetów z enum
    
    Returns:
        Lista DatasetConfig dla danej grupy
    """
    if group == DatasetGroup.STANDARD_DIGITS:
        return DIGITS_ONLY_DATASETS
    elif group == DatasetGroup.ALL_EMNIST:
        return ALL_EMNIST_DATASETS
    elif group == DatasetGroup.ALL_BASIC:
        return BASIC_DATASETS
    elif group == DatasetGroup.MNIST_C_ALL:
        return ALL_MNIST_C_DATASETS
    elif group == DatasetGroup.EVERYTHING:
        return ALL_DATASETS
    else:
        raise ValueError(f"Nieznana grupa: {group}")


def get_datasets_by_names(dataset_names: List[DatasetName],
                          mnist_c_corruptions: Optional[List[MnistCCorruption]] = None) -> List[DatasetConfig]:
    """
    Pobiera listę datasetów na podstawie listy nazw.
    
    Args:
        dataset_names: Lista nazw datasetów
        mnist_c_corruptions: Lista korupcji dla MNIST-C (jeśli MNIST_C w dataset_names)
    
    Returns:
        Lista DatasetConfig
    """
    result = []
    
    for name in dataset_names:
        if name == DatasetName.MNIST_C:
            if mnist_c_corruptions:
                for corruption in mnist_c_corruptions:
                    result.append(create_mnist_c_config(corruption))
            else:
                # Domyślnie wszystkie korupcje
                result.extend(ALL_MNIST_C_DATASETS)
        else:
            result.append(get_dataset(name))
    
    return result

