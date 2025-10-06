"""
Przykłady użycia Bayesian Optimizer - dla developerów.
"""
from Testers.BayesianOptimizer import (
    OptimizationOrchestrator,
    SearchSpaceConfig,
    FixedParamsConfig,
    DatasetConfig,
    MNIST_DATASET,
    FULL_SEARCH_SPACE
)
from Testers.Shared.DataLoader import DataType
from Testers.Shared.configs import TestRunnerConfig


def example_1_simple_usage():
    """
    Przykład 1: Najprostsze użycie - domyślna konfiguracja.
    """
    print("Example 1: Simple usage with defaults")
    
    orchestrator = OptimizationOrchestrator(
        datasets=[MNIST_DATASET],
        search_space_config=FULL_SEARCH_SPACE,
        n_iterations=10  # Mało iteracji dla przykładu
    )
    
    results = orchestrator.run_optimization()
    
    # Dostęp do wyników
    mnist_result = results['MNIST']
    print(f"Best accuracy: {mnist_result.best_accuracy:.4f}")
    print(f"Best params: {mnist_result.best_params}")


def example_2_custom_search_space():
    """
    Przykład 2: Własna przestrzeń przeszukiwania.
    """
    print("\nExample 2: Custom search space")
    
    # Zawężona przestrzeń dla szybszej optymalizacji
    custom_search_space = SearchSpaceConfig(
        learning_rate_min=0.05,
        learning_rate_max=0.15,  # Wąski zakres
        max_depth_min=5,
        max_depth_max=7,  # Tylko płytkie drzewa
        n_estimators_min=100,
        n_estimators_max=200
    )
    
    orchestrator = OptimizationOrchestrator(
        datasets=[MNIST_DATASET],
        search_space_config=custom_search_space,
        n_iterations=10
    )
    
    results = orchestrator.run_optimization()


def example_3_custom_dataset():
    """
    Przykład 3: Własny dataset.
    """
    print("\nExample 3: Custom dataset")
    
    # Stwórz własną konfigurację datasetu
    my_dataset = DatasetConfig(
        name='My Custom Dataset',
        train_path='Data/my_train.csv',
        test_path='Data/my_test.csv',
        data_type=DataType.MNIST_FORMAT,
        class_count=10
    )
    
    orchestrator = OptimizationOrchestrator(
        datasets=[my_dataset],
        search_space_config=FULL_SEARCH_SPACE,
        n_iterations=10
    )
    
    # Opcjonalnie: nie wyświetlaj szczegółowych logów
    orchestrator.verbose = False
    
    results = orchestrator.run_optimization()


def example_4_custom_test_runner_config():
    """
    Przykład 4: Własna konfiguracja test runnera.
    """
    print("\nExample 4: Custom test runner config")
    
    # Własna konfiguracja test runnera
    custom_runner_config = TestRunnerConfig(
        skip_first_vector_generation=True,  # Pomiń pierwszą generację
        save_results_after_each_test=False   # Zapisz tylko na koniec
    )
    
    orchestrator = OptimizationOrchestrator(
        datasets=[MNIST_DATASET],
        search_space_config=FULL_SEARCH_SPACE,
        n_iterations=10,
        test_runner_config=custom_runner_config
    )
    
    results = orchestrator.run_optimization()


def example_5_direct_optimizer_use():
    """
    Przykład 5: Bezpośrednie użycie BayesianOptimizer (advanced).
    """
    print("\nExample 5: Direct BayesianOptimizer usage (advanced)")
    
    from Testers.BayesianOptimizer import BayesianOptimizer
    from Testers.XgBoostTester.XGBTestRunner import XGBTestRunner
    
    # Własny test runner
    test_runner = XGBTestRunner(
        train_dataset_path='Data/mnist_train.csv',
        test_dataset_path='Data/mnist_test.csv',
        train_data_type=DataType.MNIST_FORMAT,
        test_data_type=DataType.MNIST_FORMAT,
        config=TestRunnerConfig()
    )
    
    # Własne fixed params
    fixed_params = FixedParamsConfig(class_count=10).to_dict()
    
    # Własny search space
    search_space = FULL_SEARCH_SPACE.to_search_space()
    
    # Bezpośrednie użycie optimizera
    optimizer = BayesianOptimizer(
        test_runner=test_runner,
        search_space=search_space,
        fixed_params=fixed_params,
        n_iterations=10,
        n_random_starts=3,
        verbose=True
    )
    
    result = optimizer.optimize()
    
    print(f"Best accuracy: {result.best_accuracy:.4f}")


def example_6_analyze_results():
    """
    Przykład 6: Analiza wyników po optymalizacji.
    """
    print("\nExample 6: Analyzing results")
    
    orchestrator = OptimizationOrchestrator(
        datasets=[MNIST_DATASET],
        search_space_config=FULL_SEARCH_SPACE,
        n_iterations=10
    )
    
    results = orchestrator.run_optimization()
    
    # Analiza dla MNIST
    mnist_result = results['MNIST']
    
    # Top 5 wyników
    top_5 = mnist_result.top_results(5)
    print("\nTop 5 configurations:")
    for i, config in enumerate(top_5, 1):
        print(f"{i}. Accuracy: {config['accuracy']:.4f}")
        print(f"   Learning rate: {config['learning_rate']:.4f}")
        print(f"   Max depth: {config['max_depth']}")
    
    # Wszystkie iteracje
    print(f"\nTotal iterations tested: {len(mnist_result.all_iterations)}")
    
    # Średnia accuracy
    avg_accuracy = sum(r['accuracy'] for r in mnist_result.all_iterations) / len(mnist_result.all_iterations)
    print(f"Average accuracy: {avg_accuracy:.4f}")


if __name__ == "__main__":
    # Uruchom wybrany przykład
    # Odkomentuj ten który chcesz przetestować
    
    example_1_simple_usage()
    # example_2_custom_search_space()
    # example_3_custom_dataset()
    # example_4_custom_test_runner_config()
    # example_5_direct_optimizer_use()
    # example_6_analyze_results()
