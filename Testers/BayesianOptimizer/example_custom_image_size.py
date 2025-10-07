"""
Przykład użycia BayesianOptimizer z custom image size (np. 32x32).
Demonstracja jak skonfigurować dataset o różnym rozmiarze obrazów.
"""

from Testers.BayesianOptimizer.dataset_config import DatasetConfig
from Testers.BayesianOptimizer.configs import SearchSpaceConfig, QUICK_SEARCH_SPACE
from Testers.BayesianOptimizer.orchestrator import OptimizationOrchestrator
from Testers.Shared.DataLoader import DataType


def create_custom_dataset_32x32():
    """Przykład konfiguracji datasetu 32x32"""
    return DatasetConfig(
        name='Custom-32x32',
        train_path='Data/custom_train_32x32.csv',  # Ścieżka do własnych danych
        test_path='Data/custom_test_32x32.csv',
        data_type=DataType.MNIST_FORMAT,  # Format: label,pixel1,pixel2,...,pixel1024
        class_count=10,
        image_size=32  # 32x32 = 1024 pikseli
    )


def create_custom_dataset_16x16():
    """Przykład konfiguracji datasetu 16x16"""
    return DatasetConfig(
        name='Custom-16x16',
        train_path='Data/custom_train_16x16.csv',
        test_path='Data/custom_test_16x16.csv',
        data_type=DataType.MNIST_FORMAT,  # Format: label,pixel1,pixel2,...,pixel256
        class_count=10,
        image_size=16  # 16x16 = 256 pikseli
    )


def example_run_optimization_custom_size():
    """
    Przykład uruchomienia optymalizacji dla datasetu o niestandardowym rozmiarze.
    """
    # Utwórz dataset z obrazami 32x32
    custom_dataset = create_custom_dataset_32x32()
    
    # Użyj szybkiej przestrzeni przeszukiwania
    search_space = QUICK_SEARCH_SPACE
    
    # Utwórz orchestrator
    orchestrator = OptimizationOrchestrator(
        datasets=[custom_dataset],
        search_space_config=search_space,
        n_iterations=10,  # Mała liczba dla testu
        n_random_starts=3,
        verbose=True
    )
    
    # Uruchom optymalizację
    results = orchestrator.run_optimization()
    
    # Wyświetl wyniki
    for dataset_name, result in results.items():
        print(f"\n{'='*60}")
        print(f"Najlepsze wyniki dla {dataset_name}:")
        print(f"Accuracy: {result.best_accuracy:.4f}")
        print(f"Parametry: {result.best_params}")
        print(f"{'='*60}")
    
    return results


def example_mixed_image_sizes():
    """
    Przykład optymalizacji dla wielu datasetów o różnych rozmiarach obrazów.
    """
    datasets = [
        DatasetConfig(
            name='MNIST-28x28',
            train_path='Data/mnist_train.csv',
            test_path='Data/mnist_test.csv',
            data_type=DataType.MNIST_FORMAT,
            class_count=10,
            image_size=28
        ),
        create_custom_dataset_32x32(),
        create_custom_dataset_16x16()
    ]
    
    orchestrator = OptimizationOrchestrator(
        datasets=datasets,
        search_space_config=QUICK_SEARCH_SPACE,
        n_iterations=10,
        n_random_starts=3,
        verbose=True
    )
    
    results = orchestrator.run_optimization()
    
    # Porównaj wyniki dla różnych rozmiarów obrazów
    print("\n" + "="*80)
    print("PORÓWNANIE WYNIKÓW DLA RÓŻNYCH ROZMIARÓW OBRAZÓW:")
    print("="*80)
    
    for dataset_name, result in results.items():
        print(f"\n{dataset_name}:")
        print(f"  Best Accuracy: {result.best_accuracy:.4f}")
        print(f"  Best num_segments: {result.best_params.get('num_segments', 'N/A')}")
    
    return results


if __name__ == "__main__":
    print("="*80)
    print("PRZYKŁAD UŻYCIA BAYESIAN OPTIMIZER Z CUSTOM IMAGE SIZE")
    print("="*80)
    print("\nTen przykład pokazuje jak używać BayesianOptimizer z obrazami")
    print("o różnych rozmiarach (np. 16x16, 28x28, 32x32, itd.)")
    print("\nWażne:")
    print("1. Plik CSV musi mieć format: label,pixel1,pixel2,...,pixelN")
    print("2. Liczba pikseli musi być kwadratem (16x16=256, 28x28=784, 32x32=1024)")
    print("3. image_size w DatasetConfig musi odpowiadać rzeczywistemu rozmiarowi")
    print("\nWartości pikseli powinny być w zakresie 0-255 (normalizacja odbywa się automatycznie)")
    print("="*80)
    
    # Odkomentuj poniższe linie, gdy będziesz mieć własne dane:
    # results = example_run_optimization_custom_size()
    # results = example_mixed_image_sizes()
    
    print("\nAby uruchomić przykłady, przygotuj pliki CSV i odkomentuj odpowiednie linie.")
