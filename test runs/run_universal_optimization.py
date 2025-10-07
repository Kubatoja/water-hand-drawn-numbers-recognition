"""
Bayesian Optimization - Multi-Dataset Universal Parameters

Znajduje JEDEN zestaw parametrów który działa dobrze na WSZYSTKICH datasetach.
Zamiast optymalizować osobno dla każdego datasetu, optymalizuje średnią accuracy.
"""

from typing import Dict, List
import numpy as np
from Testers.BayesianOptimizer import (
    BayesianOptimizer,
    FULL_SEARCH_SPACE,
    MNIST_DATASET,
    EMNIST_DIGITS_DATASET,
    USPS_DATASET,
    ALL_MNIST_C_DATASETS,
    DatasetConfig,
)
from Testers.XgBoostTester.XGBTestRunner import XGBTestRunner
from Testers.Shared.configs import TestRunnerConfig
from Testers.BayesianOptimizer.configs import FixedParamsConfig


class MultiDatasetTestRunner:
    """Test runner który testuje na wielu datasetach jednocześnie"""
    
    def __init__(self, datasets: List[DatasetConfig]):
        self.datasets = datasets
        self.test_runner_config = TestRunnerConfig(
            skip_first_vector_generation=False,
            save_results_after_each_test=True
        )
        
        # Utwórz test runnery dla każdego datasetu
        self.test_runners = []
        for dataset in datasets:
            runner = XGBTestRunner(
                train_dataset_path=dataset.train_path,
                test_dataset_path=dataset.test_path,
                train_data_type=dataset.data_type,
                test_data_type=dataset.data_type,
                train_labels_path=dataset.train_labels_path,
                test_labels_path=dataset.test_labels_path,
                config=self.test_runner_config
            )
            self.test_runners.append(runner)
    
    def evaluate_params(self, params: Dict) -> float:
        """
        Testuje parametry na wszystkich datasetach i zwraca średnią accuracy.
        
        Args:
            params: Słownik z parametrami do przetestowania
            
        Returns:
            Średnia accuracy ze wszystkich datasetów
        """
        accuracies = []
        
        print(f"\n{'='*80}")
        print(f"Testing params on {len(self.datasets)} datasets:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        print(f"{'='*80}")
        
        for i, (dataset, runner) in enumerate(zip(self.datasets, self.test_runners), 1):
            try:
                # Utwórz test config dla tego datasetu
                from Testers.XgBoostTester.configs import XGBTestConfig
                
                test_config = XGBTestConfig(
                    num_segments=params['num_segments'],
                    pixel_normalization_rate=params['pixel_normalization_rate'],
                    training_set_limit=params['training_set_limit'],
                    flood_config_str=params['flood_config'],
                    class_count=dataset.class_count,
                    image_size=dataset.image_size,
                    # XGBoost params - użyj domyślnych jeśli nie ma w params
                    n_estimators=params.get('n_estimators', 100),
                    max_depth=params.get('max_depth', 6),
                    learning_rate=params.get('learning_rate', 0.1),
                    subsample=params.get('subsample', 0.8),
                    colsample_bytree=params.get('colsample_bytree', 0.8),
                )
                
                # Uruchom test
                results = runner.run_tests([test_config])
                
                if results and len(results) > 0:
                    accuracy = results[0].accuracy
                    accuracies.append(accuracy)
                    print(f"  [{i}/{len(self.datasets)}] {dataset.display_name:30s} Accuracy: {accuracy:.4f}")
                else:
                    print(f"  [{i}/{len(self.datasets)}] {dataset.display_name:30s} FAILED")
                    
            except Exception as e:
                print(f"  [{i}/{len(self.datasets)}] {dataset.display_name:30s} ERROR: {e}")
        
        if not accuracies:
            return 0.0
        
        mean_accuracy = np.mean(accuracies)
        min_accuracy = np.min(accuracies)
        max_accuracy = np.max(accuracies)
        std_accuracy = np.std(accuracies)
        
        print(f"\n  Mean accuracy: {mean_accuracy:.4f}")
        print(f"  Min accuracy:  {min_accuracy:.4f}")
        print(f"  Max accuracy:  {max_accuracy:.4f}")
        print(f"  Std deviation: {std_accuracy:.4f}")
        print(f"{'='*80}\n")
        
        return mean_accuracy


def optimize_universal_params(datasets: List[DatasetConfig], 
                              n_iterations: int = 50,
                              n_random_starts: int = 10):
    """
    Optymalizuje uniwersalny zestaw parametrów dla wszystkich datasetów.
    
    Args:
        datasets: Lista datasetów do testowania
        n_iterations: Liczba iteracji optymalizacji
        n_random_starts: Liczba losowych startów
    """
    
    print("=" * 80)
    print("UNIVERSAL PARAMETER OPTIMIZATION")
    print("=" * 80)
    print(f"\nOptymalizacja JEDNEGO zestawu parametrów dla {len(datasets)} datasetów:")
    for i, ds in enumerate(datasets, 1):
        print(f"  {i:2d}. {ds.display_name}")
    
    print(f"\n{'='*80}")
    print(f"Iterations: {n_iterations}")
    print(f"Random starts: {n_random_starts}")
    print(f"Metric: Mean accuracy across all datasets")
    print("=" * 80)
    
    response = input("\nKontynuować? (y/n): ")
    if response.lower() != 'y':
        print("Anulowano")
        return None
    
    # Utwórz multi-dataset runner
    multi_runner = MultiDatasetTestRunner(datasets)
    
    # Dostosuj search space - używamy tego samego co w FULL_SEARCH_SPACE
    from Testers.BayesianOptimizer.configs import SearchSpaceConfig, ParamRange
    
    search_space = FULL_SEARCH_SPACE.to_search_space()
    
    # Utwórz fixed params - używamy wartości z pierwszego datasetu jako bazowe
    # (będą nadpisane przez optymalizację)
    fixed_params = FixedParamsConfig(
        class_count=10,  # Wszystkie nasze datasety mają 10 klas
        image_size=28    # Większość ma 28x28 (USPS 16x16 będzie skalowany)
    ).to_dict()
    
    # Utwórz wrapper funkcji objective
    def objective_function(params: Dict) -> float:
        # Dodaj fixed params
        full_params = {**fixed_params, **params}
        return multi_runner.evaluate_params(full_params)
    
    # Ręczna optymalizacja bayesowska
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    
    # Definiuj przestrzeń przeszukiwania
    space = [
        Integer(3, 10, name='num_segments'),
        Real(0.1, 0.5, name='pixel_normalization_rate'),
        Integer(10000, 60000, name='training_set_limit'),
        Categorical(['1111', '1110', '1101', '1011'], name='flood_config'),
    ]
    
    print("\n" + "=" * 80)
    print("Starting Bayesian Optimization...")
    print("=" * 80 + "\n")
    
    # Uruchom optymalizację
    result = gp_minimize(
        func=lambda x: -objective_function({
            'num_segments': x[0],
            'pixel_normalization_rate': x[1],
            'training_set_limit': x[2],
            'flood_config': x[3],
        }),  # Minus bo gp_minimize minimalizuje
        dimensions=space,
        n_calls=n_iterations,
        n_initial_points=n_random_starts,
        random_state=42,
        verbose=True
    )
    
    # Wyniki
    best_params = {
        'num_segments': result.x[0],
        'pixel_normalization_rate': result.x[1],
        'training_set_limit': result.x[2],
        'flood_config': result.x[3],
    }
    best_score = -result.fun  # Minus bo minimalizowaliśmy
    
    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)
    print(f"\nBest mean accuracy: {best_score:.4f}")
    print(f"\nBest universal parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    print("=" * 80)
    
    # Zapisz wyniki
    import json
    from pathlib import Path
    from datetime import datetime
    
    results_dir = Path("results/universal_params")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"universal_params_{timestamp}.json"
    
    results_data = {
        'timestamp': timestamp,
        'n_datasets': len(datasets),
        'datasets': [ds.display_name for ds in datasets],
        'n_iterations': n_iterations,
        'best_mean_accuracy': float(best_score),
        'best_params': best_params,
    }
    
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return best_params, best_score


def main():
    # Datasety do optymalizacji
    datasets = [
        MNIST_DATASET,
        EMNIST_DIGITS_DATASET,
        USPS_DATASET,
    ] + ALL_MNIST_C_DATASETS
    
    # Uruchom optymalizację
    best_params, best_score = optimize_universal_params(
        datasets=datasets,
        n_iterations=50,
        n_random_starts=10
    )
    
    if best_params:
        print("\n" + "=" * 80)
        print("Możesz teraz użyć tych parametrów dla wszystkich datasetów!")
        print("=" * 80)


if __name__ == "__main__":
    main()
