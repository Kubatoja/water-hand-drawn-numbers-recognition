"""
Bayesian Optimizer dla hiperparametrów XGBoost.
Odpowiedzialność: Orchestracja procesu optymalizacji.
"""
from typing import Dict, List, Tuple, Any, Callable
from dataclasses import dataclass, field
from skopt import gp_minimize
from skopt.space import Dimension
from skopt.utils import use_named_args

from Testers.XgBoostTester.XGBTestRunner import XGBTestRunner
from Testers.XgBoostTester.configs import XGBTestConfig
from Testers.Shared.models import TestResult


@dataclass
class OptimizationResult:
    """Wynik pojedynczej optymalizacji."""
    best_accuracy: float
    best_params: Dict[str, Any]
    all_iterations: List[Dict[str, Any]] = field(default_factory=list)
    dataset_name: str = ""
    
    @property
    def top_results(self, n: int = 5) -> List[Dict[str, Any]]:
        """Zwraca top N wyników."""
        return sorted(self.all_iterations, key=lambda x: x['accuracy'], reverse=True)[:n]


class BayesianOptimizer:
    """
    Optymalizator używający Bayesian Optimization do znajdowania
    optymalnych hiperparametrów XGBoost.
    """
    
    def __init__(
        self,
        test_runner: XGBTestRunner,
        search_space: List[Dimension],
        fixed_params: Dict[str, Any],
        n_iterations: int = 50,
        n_random_starts: int = 10,
        random_state: int = 42,
        verbose: bool = True
    ):
        """
        Args:
            test_runner: Runner do wykonywania testów XGBoost
            search_space: Przestrzeń przeszukiwania (lista skopt.space)
            fixed_params: Parametry stałe (np. num_segments, class_count)
            n_iterations: Liczba całkowitych iteracji
            n_random_starts: Liczba początkowych losowych prób
            random_state: Seed dla reprodukowalności
            verbose: Czy wyświetlać szczegółowe logi
        """
        self.test_runner = test_runner
        self.search_space = search_space
        self.fixed_params = fixed_params
        self.n_iterations = n_iterations
        self.n_random_starts = n_random_starts
        self.random_state = random_state
        self.verbose = verbose
        
        # Stan optymalizacji
        self._best_accuracy = 0.0
        self._best_params = {}
        self._iteration_count = 0
        self._all_results = []
    
    def optimize(self) -> OptimizationResult:
        """
        Uruchamia proces optymalizacji Bayesian.
        
        Returns:
            OptimizationResult z najlepszymi parametrami i historią
        """
        if self.verbose:
            self._print_header()
        
        # Tworzymy funkcję celu z zamknięciem
        objective_fn = self._create_objective_function()
        
        # Uruchamiamy optymalizację
        _ = gp_minimize(
            objective_fn,
            self.search_space,
            n_calls=self.n_iterations,
            n_random_starts=self.n_random_starts,
            random_state=self.random_state,
            verbose=False
        )
        
        if self.verbose:
            self._print_summary()
        
        return OptimizationResult(
            best_accuracy=self._best_accuracy,
            best_params=self._best_params,
            all_iterations=self._all_results,
            dataset_name=self.fixed_params.get('dataset_name', 'Unknown')
        )
    
    def _create_objective_function(self) -> Callable:
        """
        Tworzy funkcję celu dla optymalizatora.
        Używa zamknięcia aby uniknąć przekazywania self do skopt.
        """
        @use_named_args(self.search_space)
        def objective(**optimized_params):
            self._iteration_count += 1
            
            # Łączymy optymalizowane i stałe parametry
            config = self._build_config(optimized_params)
            
            if self.verbose:
                self._print_iteration_start(optimized_params)
            
            # Wykonaj test
            accuracy = self._run_single_test(config)
            
            # Zapisz wynik
            self._record_result(optimized_params, accuracy)
            
            if self.verbose:
                self._print_iteration_result(accuracy)
            
            # Zwróć negatywną accuracy (minimalizacja)
            return -accuracy
        
        return objective
    
    def _build_config(self, optimized_params: Dict[str, Any]) -> XGBTestConfig:
        """Buduje XGBTestConfig z optymalizowanych i stałych parametrów."""
        return XGBTestConfig(
            **optimized_params,
            **self.fixed_params
        )
    
    def _run_single_test(self, config: XGBTestConfig) -> float:
        """Uruchamia pojedynczy test i zwraca accuracy."""
        try:
            results: List[TestResult] = self.test_runner.run_tests([config])
            return results[0].accuracy if results else 0.0
        except Exception as e:
            print(f"❌ Error in test: {e}")
            return 0.0
    
    def _record_result(self, params: Dict[str, Any], accuracy: float):
        """Zapisuje wynik iteracji."""
        result_entry = {
            **params,
            'accuracy': accuracy,
            'iteration': self._iteration_count
        }
        self._all_results.append(result_entry)
        
        # Aktualizuj najlepszy wynik
        if accuracy > self._best_accuracy:
            self._best_accuracy = accuracy
            self._best_params = params.copy()
    
    # === Metody formatowania outputu ===
    
    def _print_header(self):
        """Wyświetla nagłówek optymalizacji."""
        print(f"\n{'='*80}")
        print(f"🔍 Bayesian Optimization")
        print(f"{'='*80}")
        print(f"   Iterations: {self.n_iterations}")
        print(f"   Random starts: {self.n_random_starts}")
        print(f"   Search space dimensions: {len(self.search_space)}")
        print(f"{'='*80}\n")
    
    def _print_iteration_start(self, params: Dict[str, Any]):
        """Wyświetla start iteracji."""
        print(f"\n🧪 Iteration {self._iteration_count}/{self.n_iterations}")
        for key, value in params.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.3f}")
            else:
                print(f"   {key}: {value}")
    
    def _print_iteration_result(self, accuracy: float):
        """Wyświetla wynik iteracji."""
        if accuracy > self._best_accuracy:
            print(f"   ✨ NEW BEST! Accuracy: {accuracy:.4f}")
        else:
            print(f"   📊 Accuracy: {accuracy:.4f} (Best: {self._best_accuracy:.4f})")
    
    def _print_summary(self):
        """Wyświetla podsumowanie optymalizacji."""
        print(f"\n{'='*80}")
        print(f"✅ OPTIMIZATION COMPLETED")
        print(f"{'='*80}")
        print(f"\n🏆 BEST PARAMETERS:")
        print(f"   Accuracy: {self._best_accuracy:.4f}")
        
        for key, value in self._best_params.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.4f}")
            else:
                print(f"   {key}: {value}")
        
        print(f"\n📊 Total iterations: {self._iteration_count}")
        
        # Top 5
        top_5 = sorted(self._all_results, key=lambda x: x['accuracy'], reverse=True)[:5]
        print(f"\n🥇 TOP 5 RESULTS:")
        for i, result in enumerate(top_5, 1):
            print(f"   {i}. Accuracy: {result['accuracy']:.4f} | Iteration: {result['iteration']}")
