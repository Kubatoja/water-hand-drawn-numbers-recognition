"""
Skrypt testowy do weryfikacji działania preprocessing'u centrowania cyfr
"""
import numpy as np
import matplotlib.pyplot as plt
from Preprocessing.image_preprocessor import ImagePreprocessor
from Tester.DataLoader import DataLoader, DataType

def test_centering():
    """Test podstawowy centrowania"""
    preprocessor = ImagePreprocessor()
    
    # Stwórzmy testowy obraz z cyfrą przesunięcą w lewo górny róg
    test_image = np.zeros((28, 28))
    # Dodajmy prostą cyfrę w lewym górnym rogu
    test_image[2:8, 2:8] = 1  # Mały kwadrat w lewym górnym rogu
    
    print("Oryginalny obraz:")
    print("Stats przed centrowaniem:", preprocessor.get_centering_stats(test_image))
    
    # Wycentruj
    centered = preprocessor.center_digit(test_image)
    
    print("\nPo centrowaniu:")
    print("Stats po centrowaniu:", preprocessor.get_centering_stats(centered))
    
    return test_image, centered

def test_real_data():
    """Test na prawdziwych danych"""
    # Wczytaj kilka próbek z EMNIST
    data_loader = DataLoader()
    
    # Użyj małego pliku testowego
    test_file = 'Data/emnist-balanced-test.csv'
    try:
        raw_data = data_loader.load_data(test_file, DataType.MNIST_FORMAT)
        print(f"Wczytano {len(raw_data)} próbek")
        
        # Sprawdź pierwsze 3 próbki
        preprocessor = ImagePreprocessor()
        
        for i in range(min(3, len(raw_data))):
            sample = raw_data[i]
            print(f"\n--- Próbka {i+1} (label: {sample.label}) ---")
            
            # Binaryzuj
            original_pixels = sample.pixels.reshape(28, 28)
            binary_image = np.where(original_pixels > 0.34, 1, 0)
            
            print("Przed centrowaniem:")
            stats_before = preprocessor.get_centering_stats(binary_image)
            print(f"  Bounding box: {stats_before['bbox']}")
            print(f"  Rozmiar: {stats_before['width']}x{stats_before['height']}")
            print(f"  Środek masy: {stats_before['center_of_mass']}")
            
            # Wycentruj
            centered = preprocessor.center_digit(binary_image)
            
            print("Po centrowaniu:")
            stats_after = preprocessor.get_centering_stats(centered)
            print(f"  Bounding box: {stats_after['bbox']}")
            print(f"  Rozmiar: {stats_after['width']}x{stats_after['height']}")
            print(f"  Środek masy: {stats_after['center_of_mass']}")
            
    except FileNotFoundError:
        print(f"Plik {test_file} nie istnieje. Sprawdź ścieżkę.")

if __name__ == "__main__":
    print("=== Test podstawowy ===")
    original, centered = test_centering()
    
    print("\n=== Test na prawdziwych danych ===")
    test_real_data()
