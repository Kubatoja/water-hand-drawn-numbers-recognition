"""
Result Reporter - formatowanie i wyświetlanie wyników optymalizacji.
Odpowiedzialność: Prezentacja wyników w czytelny sposób.
"""
from typing import Dict, List
from Testers.BayesianOptimizer.BayesianOptimizer import OptimizationResult


class ResultReporter:
    """Raportowanie wyników optymalizacji."""
    
    @staticmethod
    def print_dataset_summary(result: OptimizationResult):
        """Wyświetla podsumowanie dla datasetu."""
        print(f"\n{'='*80}")
        print(f"📊 Dataset: {result.dataset_name}")
        print(f"{'='*80}")
        print(f"🏆 Best Accuracy: {result.best_accuracy:.4f}")
        print(f"\n📋 Best Parameters:")
        
        ResultReporter._print_params(result.best_params)
        
        print(f"\n🥇 Top 5 Results:")
        for i, res in enumerate(result.top_results(5), 1):
            print(f"   {i}. Accuracy: {res['accuracy']:.4f} "
                  f"(Iteration {res['iteration']})")
    
    @staticmethod
    def print_final_summary(results: Dict[str, OptimizationResult]):
        """Wyświetla finalne podsumowanie wszystkich datasetów."""
        print(f"\n{'='*80}")
        print("🎉 ALL OPTIMIZATIONS COMPLETED!")
        print(f"{'='*80}\n")
        
        for dataset_name, result in results.items():
            print(f"📊 {dataset_name}:")
            print(f"   Best Accuracy: {result.best_accuracy:.4f}")
            print(f"   Total iterations: {len(result.all_iterations)}")
            print()
    
    @staticmethod
    def _print_params(params: Dict):
        """Pomocnicza metoda do drukowania parametrów."""
        for key, value in params.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.4f}")
            else:
                print(f"   {key}: {value}")
