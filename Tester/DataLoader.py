from dataclasses import dataclass, field
from enum import Enum
from typing import List
from pathlib import Path
import csv
import numpy as np

from Tester.otherModels import RawNumberData


class DataType(Enum):
    """Enum określający typ danych do wczytania"""
    MNIST_FORMAT = "mnist_format"

class DataLoader:
    """Klasa do wczytywania różnych typów danych"""

    def __init__(self):
        """Inicjalizuje DataLoader"""
        self._loaders = {
            DataType.MNIST_FORMAT: self._load_mnist,
        }

    def load_data(self, file_path: str, data_type: DataType) -> List[RawNumberData]:
        """
        Główna metoda do wczytywania danych

        Args:
            file_path: Ścieżka do pliku z danymi
            data_type: Typ danych do wczytania

        Returns:
            Lista obiektów RawNumberData

        Raises:
            ValueError: Jeśli nieobsługiwany typ danych
            FileNotFoundError: Jeśli plik nie istnieje
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if data_type not in self._loaders:
            raise ValueError(f"Unsupported data type: {data_type}")

        print(f"Loading {data_type.value} data from {file_path}...")
        return self._loaders[data_type](file_path)

    def _load_mnist(self, file_path: str) -> List[RawNumberData]:
        """
        Wczytuje surowe dane liczb (oryginalny format)
        Format: label, pixel1, pixel2, ..., pixelN
        """
        data = []
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                data.append(row)

        data = np.array(data, dtype=float)
        labels = data[:, 0]  # Pierwsza kolumna to etykieta
        pixels = data[:, 1:] / 255.0  # Normalizacja pikseli do zakresu [0, 1]

        # Tworzenie listy obiektów RawNumberData
        result = []
        for i in range(len(labels)):
            raw_number = RawNumberData(
                label=int(labels[i]),
                pixels=pixels[i]
            )
            result.append(raw_number)

        print(f"Loaded {len(result)} raw number samples")
        return result