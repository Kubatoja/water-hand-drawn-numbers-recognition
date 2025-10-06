"""
Dataset Configuration dla Bayesian Optimization.
Odpowiedzialność: Konfiguracja datasetów do testowania.
"""
from dataclasses import dataclass
from typing import List

from Testers.Shared.DataLoader import DataType


@dataclass
class DatasetConfig:
    """Konfiguracja pojedynczego datasetu."""
    
    name: str
    train_path: str
    test_path: str
    data_type: DataType
    class_count: int
    
    def __str__(self) -> str:
        return f"{self.name} ({self.class_count} classes)"


# Predefiniowane datasety
MNIST_DATASET = DatasetConfig(
    name='MNIST',
    train_path='Data/mnist_train.csv',
    test_path='Data/mnist_test.csv',
    data_type=DataType.MNIST_FORMAT,
    class_count=10
)

EMNIST_DATASET = DatasetConfig(
    name='EMNIST-Balanced',
    train_path='Data/emnist-balanced-train.csv',
    test_path='Data/emnist-balanced-test.csv',
    data_type=DataType.EMNIST_FORMAT,
    class_count=47
)

ALL_DATASETS: List[DatasetConfig] = [MNIST_DATASET, EMNIST_DATASET]
