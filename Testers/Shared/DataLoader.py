from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Optional
from pathlib import Path
import csv
import numpy as np
import bz2

from Testers.Shared.models import RawNumberData


class DataType(Enum):
    """Enum określający typ danych do wczytania"""
    MNIST_FORMAT = "mnist_format"           # label,pixel1,pixel2,... (standard CSV)
    SEPARATED_FORMAT = "separated"          # Oddzielne pliki dla obrazów i etykiet (Arabic)
    LIBSVM_FORMAT = "libsvm"               # Format LibSVM (USPS)
    NPY_FORMAT = "npy"                     # NumPy arrays (MNIST-C)


class DataLoader:
    """Klasa do wczytywania różnych typów danych"""

    def __init__(self, project_root: Optional[str] = None):
        """
        Inicjalizuje DataLoader
        
        Args:
            project_root: Główny katalog projektu. Jeśli None, wykrywa automatycznie.
        """
        self._loaders = {
            DataType.MNIST_FORMAT: self._load_mnist,
            DataType.SEPARATED_FORMAT: self._load_separated,
            DataType.LIBSVM_FORMAT: self._load_libsvm,
            DataType.NPY_FORMAT: self._load_npy,
        }
        
        # Wykryj katalog główny projektu (tam gdzie jest folder Data/)
        if project_root:
            self.project_root = Path(project_root)
        else:
            # Szukaj katalogu głównego projektu (zawierającego folder Data/)
            current = Path(__file__).resolve().parent
            while current.parent != current:
                if (current / "Data").exists():
                    self.project_root = current
                    break
                current = current.parent
            else:
                # Fallback: użyj bieżącego katalogu
                self.project_root = Path.cwd()

    def load_data(self, file_path: str, data_type: DataType, 
                  labels_path: Optional[str] = None,
                  is_training_data: bool = True) -> List[RawNumberData]:
        """
        Główna metoda do wczytywania danych

        Args:
            file_path: Ścieżka do pliku z danymi (lub folderu dla NPY)
            data_type: Typ danych do wczytania
            labels_path: Opcjonalna ścieżka do pliku z etykietami (dla SEPARATED_FORMAT)
            is_training_data: True dla danych treningowych, False dla testowych (używane dla NPY_FORMAT)

        Returns:
            Lista obiektów RawNumberData

        Raises:
            ValueError: Jeśli nieobsługiwany typ danych
            FileNotFoundError: Jeśli plik nie istnieje
        """
        # Konwertuj ścieżkę względną na bezwzględną (względem project_root)
        path_obj = Path(file_path)
        if not path_obj.is_absolute():
            path_obj = self.project_root / file_path
        
        # Walidacja ścieżki (plik lub folder dla NPY)
        if data_type == DataType.NPY_FORMAT:
            if not path_obj.is_dir():
                raise FileNotFoundError(f"Directory not found: {path_obj}")
        else:
            if not path_obj.exists():
                raise FileNotFoundError(f"File not found: {path_obj}")

        if data_type not in self._loaders:
            raise ValueError(f"Unsupported data type: {data_type}")

        print(f"Loading {data_type.value} data from {path_obj}...")
        
        # Przekaż labels_path jeśli loader go wymaga
        if data_type == DataType.SEPARATED_FORMAT:
            if not labels_path:
                raise ValueError(f"labels_path is required for {data_type}")
            # Konwertuj również labels_path
            labels_path_obj = Path(labels_path)
            if not labels_path_obj.is_absolute():
                labels_path_obj = self.project_root / labels_path
            return self._loaders[data_type](str(path_obj), str(labels_path_obj))
        elif data_type == DataType.NPY_FORMAT:
            # Dla NPY przekaż informację czy to dane treningowe czy testowe
            return self._loaders[data_type](str(path_obj), is_training_data)
        else:
            return self._loaders[data_type](str(path_obj))

    def _load_mnist(self, file_path: str) -> List[RawNumberData]:
        """
        Wczytuje surowe dane liczb (oryginalny format)
        Format: label, pixel1, pixel2, ..., pixelN
        Automatycznie wykrywa rozmiar obrazu na podstawie liczby pikseli.
        Pomija pierwszy wiersz jeśli zawiera nagłówek (np. 'label', 'pixel1', ...).
        """
        data = []
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            first_row = True
            for row in reader:
                # Sprawdź czy pierwszy wiersz to nagłówek
                if first_row:
                    first_row = False
                    # Jeśli pierwsza wartość to 'label' lub podobny tekst, pomiń
                    try:
                        float(row[0])
                        # To są dane, nie nagłówek - dodaj
                        data.append(row)
                    except ValueError:
                        # To nagłówek - pomiń
                        continue
                else:
                    data.append(row)

        data = np.array(data, dtype=float)
        labels = data[:, 0]  # Pierwsza kolumna to etykieta
        pixels = data[:, 1:] / 255.0  # Normalizacja pikseli do zakresu [0, 1]

        # Automatyczne wykrycie rozmiaru obrazu
        num_pixels = pixels.shape[1]
        image_size = int(np.sqrt(num_pixels))
        if image_size * image_size != num_pixels:
            raise ValueError(f"Liczba pikseli ({num_pixels}) musi być kwadratem (np. 784=28x28, 1024=32x32)")

        # Tworzenie listy obiektów RawNumberData
        result = []
        for i in range(len(labels)):
            raw_number = RawNumberData(
                label=int(labels[i]),
                pixels=pixels[i]
            )
            result.append(raw_number)

        print(f"Loaded {len(result)} raw number samples (image size: {image_size}x{image_size})")
        return result

    def _load_separated(self, images_path: str, labels_path: str) -> List[RawNumberData]:
        """
        Wczytuje dane z oddzielnych plików dla obrazów i etykiet (format Arabic)
        
        Args:
            images_path: Ścieżka do pliku CSV z obrazami (N x pixels)
            labels_path: Ścieżka do pliku CSV z etykietami (N x 1)
        """
        # Wczytaj obrazy
        images = []
        with open(images_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                images.append(row)
        
        images = np.array(images, dtype=float) / 255.0  # Normalizacja
        
        # Wczytaj etykiety
        labels = []
        with open(labels_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                labels.append(row[0])
        
        labels = np.array(labels, dtype=int)
        
        if len(images) != len(labels):
            raise ValueError(f"Mismatch: {len(images)} images vs {len(labels)} labels")
        
        # Wykryj rozmiar obrazu
        num_pixels = images.shape[1]
        image_size = int(np.sqrt(num_pixels))
        if image_size * image_size != num_pixels:
            raise ValueError(f"Liczba pikseli ({num_pixels}) musi być kwadratem")
        
        # Tworzenie listy obiektów
        result = []
        for i in range(len(labels)):
            raw_number = RawNumberData(
                label=int(labels[i]),
                pixels=images[i]
            )
            result.append(raw_number)
        
        print(f"Loaded {len(result)} raw number samples from separated files (image size: {image_size}x{image_size})")
        return result

    def _load_libsvm(self, file_path: str) -> List[RawNumberData]:
        """
        Wczytuje dane w formacie LibSVM (format USPS)
        Format: label feature1:value1 feature2:value2 ...
        
        Obsługuje również pliki skompresowane .bz2
        """
        # Sprawdź czy plik jest skompresowany
        is_compressed = file_path.endswith('.bz2')
        
        # Otwórz plik
        if is_compressed:
            file_handle = bz2.open(file_path, 'rt')
        else:
            file_handle = open(file_path, 'r')
        
        try:
            labels = []
            pixels_list = []
            max_feature_idx = 0
            
            for line in file_handle:
                parts = line.strip().split()
                if not parts:
                    continue
                
                # Pierwsza część to etykieta
                # USPS używa etykiet 1-10, konwertujemy do 0-9
                label = int(float(parts[0])) - 1
                labels.append(label)
                
                # Pozostałe to pary feature:value
                features = {}
                for part in parts[1:]:
                    if ':' in part:
                        idx, value = part.split(':')
                        idx = int(idx)
                        features[idx] = float(value)
                        max_feature_idx = max(max_feature_idx, idx)
                
                pixels_list.append(features)
            
            # Konwertuj sparse features do dense array
            num_features = max_feature_idx
            dense_pixels = np.zeros((len(labels), num_features))
            
            for i, features in enumerate(pixels_list):
                for idx, value in features.items():
                    dense_pixels[i, idx - 1] = value  # LibSVM używa 1-based indexing
            
            # Normalizacja: LibSVM USPS używa zakresu [-1, 1], konwertujemy do [0, 1]
            dense_pixels = (dense_pixels + 1.0) / 2.0
            
            # Wykryj rozmiar obrazu
            image_size = int(np.sqrt(num_features))
            if image_size * image_size != num_features:
                # USPS ma 256 features (16x16)
                print(f"Warning: {num_features} features - może nie być kwadratowy obraz")
                image_size = 16 if num_features == 256 else int(np.sqrt(num_features))
            
            # Tworzenie listy obiektów
            result = []
            for i in range(len(labels)):
                raw_number = RawNumberData(
                    label=labels[i],
                    pixels=dense_pixels[i]
                )
                result.append(raw_number)
            
            print(f"Loaded {len(result)} raw number samples from LibSVM format (image size: {image_size}x{image_size})")
            return result
            
        finally:
            file_handle.close()

    def _load_npy(self, folder_path: str, is_training_data: bool = True) -> List[RawNumberData]:
        """
        Wczytuje dane z plików NumPy (.npy) - format MNIST-C
        
        Oczekuje struktury:
        folder_path/
            train_images.npy  (N, H, W) lub (N, H, W, 1)
            train_labels.npy  (N,)
            test_images.npy   (N, H, W) lub (N, H, W, 1)
            test_labels.npy   (N,)
        
        Args:
            folder_path: Ścieżka do folderu z plikami .npy
            is_training_data: True dla danych treningowych, False dla testowych
        """
        folder = Path(folder_path)
        
        # Wybierz odpowiednie pliki na podstawie parametru
        if is_training_data:
            images_file = folder / 'train_images.npy'
            labels_file = folder / 'train_labels.npy'
            data_type_str = "training"
        else:
            images_file = folder / 'test_images.npy'
            labels_file = folder / 'test_labels.npy'
            data_type_str = "test"
        
        if not images_file.exists():
            raise FileNotFoundError(f"Images file not found: {images_file}")
        if not labels_file.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_file}")
        
        # Wczytaj dane
        images = np.load(images_file)
        labels = np.load(labels_file)
        
        if len(images) != len(labels):
            raise ValueError(f"Mismatch: {len(images)} images vs {len(labels)} labels")
        
        # Obsługa różnych kształtów: (N, H, W, 1) lub (N, H, W)
        if len(images.shape) == 4 and images.shape[-1] == 1:
            images = images.squeeze(-1)  # Usuń ostatni wymiar
        
        if len(images.shape) != 3:
            raise ValueError(f"Expected 3D array (N, H, W), got shape {images.shape}")
        
        num_samples, height, width = images.shape
        
        # Normalizacja do [0, 1]
        images = images.astype(float) / 255.0
        
        # Spłaszcz obrazy do 1D
        pixels = images.reshape(num_samples, -1)
        
        # Tworzenie listy obiektów
        result = []
        for i in range(len(labels)):
            raw_number = RawNumberData(
                label=int(labels[i]),
                pixels=pixels[i]
            )
            result.append(raw_number)
        
        print(f"Loaded {len(result)} {data_type_str} samples from NPY format (image size: {height}x{width})")
        return result
