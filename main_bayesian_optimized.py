"""
Main entry point for Bayesian Optimization of XGBoost hyperparameters.

Użycie:
    python main_bayesian_optimized.py              # Pełna optymalizacja (50 iteracji)
    python main_bayesian_optimized.py --quick      # Szybki test (10 iteracji)
    python main_bayesian_optimized.py --mnist      # Tylko MNIST
"""
import argparse
from Testers.BayesianOptimizer import (
    OptimizationOrchestrator,
    FULL_SEARCH_SPACE,
    QUICK_SEARCH_SPACE,
    ALL_DATASETS,
    MNIST_DATASET,
    EMNIST_DATASET
)


def parse_args():
    """Parsowanie argumentów linii poleceń."""
    parser = argparse.ArgumentParser(
        description='Bayesian Optimization for XGBoost hyperparameters'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick test mode (10 iterations, reduced search space)'
    )
    
    parser.add_argument(
        '--mnist',
        action='store_true',
        help='Optimize only for MNIST dataset'
    )
    
    parser.add_argument(
        '--emnist',
        action='store_true',
        help='Optimize only for EMNIST dataset'
    )
    
    parser.add_argument(
        '--iterations',
        type=int,
        default=None,
        help='Custom number of iterations (overrides --quick)'
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()
    
    # Determine datasets to use
    if args.mnist:
        datasets = [MNIST_DATASET]
    elif args.emnist:
        datasets = [EMNIST_DATASET]
    else:
        datasets = ALL_DATASETS
    
    # Determine search space and iterations
    if args.quick and args.iterations is None:
        search_space = QUICK_SEARCH_SPACE
        n_iterations = 10
        n_random_starts = 3
    else:
        search_space = FULL_SEARCH_SPACE
        n_iterations = args.iterations if args.iterations else 50
        n_random_starts = max(3, n_iterations // 5)  # 20% random starts
    
    # Create and run orchestrator
    orchestrator = OptimizationOrchestrator(
        datasets=datasets,
        search_space_config=search_space,
        n_iterations=n_iterations,
        n_random_starts=n_random_starts,
        verbose=True
    )
    
    # Run optimization
    results = orchestrator.run_optimization()
    
    # Results are automatically saved by XGBTestRunner
    print("\n✅ Optimization completed. Results saved to results/ directory.")


if __name__ == "__main__":
    main()
