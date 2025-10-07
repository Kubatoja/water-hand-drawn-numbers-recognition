"""
Loader dla MNIST-C (Corrupted MNIST) dataset.
Dataset zawiera obrazy MNIST z różnymi typami zniekształceń.

Źródło: https://github.com/google-research/mnist-c
Dataset można pobrać przez tensorflow_datasets lub bezpośrednio z repozytorium.
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple
import csv

from Testers.Shared.models import RawNumberData


class MNISTCorruptedLoader:
    """
    Klasa do wczytywania MNIST-C dataset.
    
    MNIST-C zawiera 15 typów korupcji:
    - shot_noise
    - impulse_noise
    - glass_blur
    - motion_blur
    - shear
    - scale
    - rotate
    - brightness
    - translate
    - stripe
    - fog
    - spatter
    - dotted_line
    - zigzag
    - canny_edges
    """
    
    CORRUPTION_TYPES = [
        'shot_noise', 'impulse_noise', 'glass_blur', 'motion_blur',
        'shear', 'scale', 'rotate', 'brightness', 'translate',
        'stripe', 'fog', 'spatter', 'dotted_line', 'zigzag', 'canny_edges'
    ]
    
    def __init__(self, mnist_c_path: str = 'Data/mnist_c'):
        """
        Args:
            mnist_c_path: Ścieżka do katalogu z danymi MNIST-C
        """
        self.mnist_c_path = Path(mnist_c_path)
    
    @staticmethod
    def load_from_tensorflow_datasets(
        corruption_type: str = 'shot_noise',
        severity: int = 1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Wczytuje MNIST-C używając tensorflow_datasets.
        
        Args:
            corruption_type: Typ korupcji (np. 'shot_noise', 'rotate', etc.)
            severity: Poziom korupcji (1-5, gdzie 5 to najgorszy)
        
        Returns:
            Tuple: (train_images, train_labels, test_images, test_labels)
            
        Requires:
            pip install tensorflow-datasets
        """
        try:
            import tensorflow_datasets as tfds
        except ImportError:
            raise ImportError(
                "tensorflow_datasets is required. Install it with:\n"
                "pip install tensorflow-datasets"
            )
        
        print(f"Loading MNIST-C from TensorFlow Datasets...")
        print(f"Corruption: {corruption_type}, Severity: {severity}")
        
        # Wczytaj dataset
        ds_train = tfds.load(
            'mnist_corrupted',
            split='train',
            as_supervised=True,
            with_info=False
        )
        
        ds_test = tfds.load(
            f'mnist_corrupted/{corruption_type}_{severity}',
            split='test',
            as_supervised=True,
            with_info=False
        )
        
        # Konwertuj do numpy
        train_images = []
        train_labels = []
        for image, label in ds_train:
            train_images.append(image.numpy())
            train_labels.append(label.numpy())
        
        test_images = []
        test_labels = []
        for image, label in ds_test:
            test_images.append(image.numpy())
            test_labels.append(label.numpy())
        
        train_images = np.array(train_images)
        train_labels = np.array(train_labels)
        test_images = np.array(test_images)
        test_labels = np.array(test_labels)
        
        print(f"Loaded {len(train_images)} training samples")
        print(f"Loaded {len(test_images)} test samples")
        print(f"Image shape: {train_images.shape[1:]}")
        
        return train_images, train_labels, test_images, test_labels
    
    @staticmethod
    def convert_to_raw_number_data(
        images: np.ndarray,
        labels: np.ndarray
    ) -> List[RawNumberData]:
        """
        Konwertuje numpy arrays do formatu RawNumberData.
        
        Args:
            images: Array z obrazami (N, H, W) lub (N, H, W, C)
            labels: Array z etykietami (N,)
        
        Returns:
            Lista obiektów RawNumberData
        """
        result = []
        
        # Jeśli obrazy są RGB/RGBA, konwertuj na grayscale
        if len(images.shape) == 4:
            # Konwersja RGB do grayscale: 0.299*R + 0.587*G + 0.114*B
            images = np.dot(images[..., :3], [0.299, 0.587, 0.114])
        
        # Normalizacja do [0, 1]
        if images.max() > 1.0:
            images = images / 255.0
        
        for i in range(len(labels)):
            # Spłaszcz obraz do 1D array
            pixels = images[i].flatten()
            
            raw_number = RawNumberData(
                label=int(labels[i]),
                pixels=pixels
            )
            result.append(raw_number)
        
        return result
    
    @staticmethod
    def save_to_csv(
        images: np.ndarray,
        labels: np.ndarray,
        output_path: str
    ):
        """
        Zapisuje dane w formacie CSV kompatybilnym z istniejącym loaderem.
        Format: label,pixel1,pixel2,...,pixelN
        
        Args:
            images: Array z obrazami (N, H, W)
            labels: Array z etykietami (N,)
            output_path: Ścieżka do pliku wyjściowego CSV
        """
        print(f"Saving to {output_path}...")
        
        # Jeśli obrazy są RGB, konwertuj na grayscale
        if len(images.shape) == 4:
            images = np.dot(images[..., :3], [0.299, 0.587, 0.114])
        
        # Denormalizuj jeśli potrzeba (do 0-255)
        if images.max() <= 1.0:
            images = (images * 255).astype(np.uint8)
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            for i in range(len(labels)):
                # Spłaszcz obraz i dodaj label na początku
                row = [int(labels[i])] + images[i].flatten().tolist()
                writer.writerow(row)
        
        print(f"Saved {len(labels)} samples to {output_path}")


def download_and_prepare_mnist_c(
    corruption_types: List[str] = None,
    severities: List[int] = None,
    output_dir: str = 'Data'
):
    """
    Pobiera i przygotowuje MNIST-C dataset w formacie CSV.
    
    Args:
        corruption_types: Lista typów korupcji (None = wszystkie)
        severities: Lista poziomów korupcji (None = [1,2,3,4,5])
        output_dir: Katalog wyjściowy dla plików CSV
    """
    if corruption_types is None:
        corruption_types = MNISTCorruptedLoader.CORRUPTION_TYPES
    
    if severities is None:
        severities = [1, 2, 3, 4, 5]
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    loader = MNISTCorruptedLoader()
    
    # Wczytaj raz dane treningowe (bez korupcji)
    print("\nLoading clean training data...")
    train_images, train_labels, _, _ = loader.load_from_tensorflow_datasets(
        corruption_type='shot_noise',  # Nieważne dla train
        severity=1
    )
    
    # Zapisz dane treningowe (czyste)
    train_csv = output_path / 'mnist_c_train_clean.csv'
    loader.save_to_csv(train_images, train_labels, str(train_csv))
    
    # Przetwórz wszystkie kombinacje korupcji i poziomów dla danych testowych
    for corruption in corruption_types:
        for severity in severities:
            print(f"\n{'='*60}")
            print(f"Processing: {corruption} (severity {severity})")
            print('='*60)
            
            try:
                _, _, test_images, test_labels = loader.load_from_tensorflow_datasets(
                    corruption_type=corruption,
                    severity=severity
                )
                
                # Zapisz do CSV
                test_csv = output_path / f'mnist_c_test_{corruption}_s{severity}.csv'
                loader.save_to_csv(test_images, test_labels, str(test_csv))
                
            except Exception as e:
                print(f"Error processing {corruption} severity {severity}: {e}")
                continue
    
    print(f"\n{'='*60}")
    print("Download and conversion completed!")
    print(f"Files saved to: {output_path}")
    print('='*60)


# Przykład użycia
if __name__ == "__main__":
    print("="*80)
    print("MNIST-C (Corrupted MNIST) Loader")
    print("="*80)
    print("\nKrok 1: Zainstaluj wymagane pakiety:")
    print("pip install tensorflow-datasets")
    print("\nKrok 2: Uruchom ten skrypt, aby pobrać dane:")
    print("python mnist_corrupted_loader.py")
    print("\nKrok 3: Użyj wygenerowanych plików CSV z BayesianOptimizer")
    print("="*80)
    
    # Przykład 1: Szybki test - jeden typ korupcji
    print("\n\nPrzykład 1: Pobierz tylko 'shot_noise' severity 1-3")
    print("Odkomentuj poniższą linię aby uruchomić:")
    print("# download_and_prepare_mnist_c(corruption_types=['shot_noise'], severities=[1,2,3])")
    
    # Przykład 2: Pełny dataset
    print("\n\nPrzykład 2: Pobierz wszystkie typy korupcji")
    print("UWAGA: To może zająć dużo czasu i miejsca!")
    print("Odkomentuj poniższą linię aby uruchomić:")
    print("# download_and_prepare_mnist_c()")
    
    # Przykład 3: Bezpośrednie użycie
    print("\n\nPrzykład 3: Bezpośrednie użycie z kodem:")
    print("""
    from mnist_corrupted_loader import MNISTCorruptedLoader
    
    loader = MNISTCorruptedLoader()
    
    # Wczytaj dane
    train_imgs, train_lbls, test_imgs, test_lbls = \\
        loader.load_from_tensorflow_datasets('rotate', severity=3)
    
    # Konwertuj do RawNumberData
    train_data = loader.convert_to_raw_number_data(train_imgs, train_lbls)
    test_data = loader.convert_to_raw_number_data(test_imgs, test_lbls)
    
    # Użyj w swoim kodzie
    """)
