"""
Orchestrator dla procesu Bayesian Optimization.
Odpowiedzialność: Koordynacja optymalizacji dla wielu datasetów.
"""
from typing import Dict, List

from Testers.XgBoostTester.XGBTestRunner import XGBTestRunner
from Testers.Shared.configs import TestRunnerConfig
from Testers.BayesianOptimizer.BayesianOptimizer import BayesianOptimizer, OptimizationResult
from Testers.BayesianOptimizer.configs import SearchSpaceConfig, FixedParamsConfig
from Testers.BayesianOptimizer.dataset_config import DatasetConfig
from Testers.BayesianOptimizer.reporter import ResultReporter


class OptimizationOrchestrator:
    """
    Orkiestrator zarządzający procesem optymalizacji dla wielu datasetów.
    """
    
    def __init__(
        self,
        datasets: List[DatasetConfig],
        search_space_config: SearchSpaceConfig,
        n_iterations: int = 50,
        n_random_starts: int = 10,
        test_runner_config: TestRunnerConfig = None,
        verbose: bool = True
    ):
        """
        Args:
            datasets: Lista datasetów do przetestowania
            search_space_config: Konfiguracja przestrzeni przeszukiwania
            n_iterations: Liczba iteracji optymalizacji
            n_random_starts: Liczba losowych startów
            test_runner_config: Konfiguracja test runnera
            verbose: Czy wyświetlać szczegółowe logi
        """
        self.datasets = datasets
        self.search_space_config = search_space_config
        self.n_iterations = n_iterations
        self.n_random_starts = n_random_starts
        self.verbose = verbose
        
        self.test_runner_config = test_runner_config or TestRunnerConfig(
            skip_first_vector_generation=False,
            save_results_after_each_test=True
        )
        
        self.reporter = ResultReporter()
        self.results: Dict[str, OptimizationResult] = {}
    
    def run_optimization(self) -> Dict[str, OptimizationResult]:
        """
        Uruchamia optymalizację dla wszystkich datasetów.
        
        Returns:
            Słownik z wynikami optymalizacji dla każdego datasetu
        """
        for dataset in self.datasets:
            result = self._optimize_single_dataset(dataset)
            self.results[dataset.name] = result
            
            if self.verbose:
                self.reporter.print_dataset_summary(result)
        
        if self.verbose:
            self.reporter.print_final_summary(self.results)
        
        return self.results
    
    def _optimize_single_dataset(self, dataset: DatasetConfig) -> OptimizationResult:
        """Optymalizuje hiperparametry dla pojedynczego datasetu."""
        # Utwórz test runner
        test_runner = self._create_test_runner(dataset)
        
        # Utwórz fixed params
        fixed_params = self._create_fixed_params(dataset)
        
        # Utwórz search space
        search_space = self.search_space_config.to_search_space()
        
        # Utwórz optymalizator
        optimizer = BayesianOptimizer(
            test_runner=test_runner,
            search_space=search_space,
            fixed_params=fixed_params,
            n_iterations=self.n_iterations,
            n_random_starts=self.n_random_starts,
            verbose=self.verbose
        )
        
        # Uruchom optymalizację
        return optimizer.optimize()
    
    def _create_test_runner(self, dataset: DatasetConfig) -> XGBTestRunner:
        """Tworzy test runner dla datasetu."""
        return XGBTestRunner(
            train_dataset_path=dataset.train_path,
            test_dataset_path=dataset.test_path,
            train_data_type=dataset.data_type,
            test_data_type=dataset.data_type,
            train_labels_path=dataset.train_labels_path,
            test_labels_path=dataset.test_labels_path,
            config=self.test_runner_config
        )
    
    def _create_fixed_params(self, dataset: DatasetConfig) -> Dict:
        """Tworzy słownik stałych parametrów (bez dataset_name)."""
        fixed_config = FixedParamsConfig(
            class_count=dataset.class_count,
            image_size=dataset.image_size
        )
        return fixed_config.to_dict()
