"""
Quick test Bayesian Optimization - tylko MNIST, mało iteracji
"""
from Testers.XgBoostTester.XGBTestRunner import XGBTestRunner
from Testers.Shared.DataLoader import DataType
from Testers.Shared.configs import TestRunnerConfig, FloodConfig
from Testers.XgBoostTester.configs import XGBTestConfig
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

def quick_test():
    """Szybki test Bayesian Optimization - tylko 10 iteracji"""
    
    print("🚀 Quick Bayesian Optimization Test")
    print("="*60)
    
    # Konfiguracja
    test_runner_config = TestRunnerConfig(
        skip_first_vector_generation=False, 
        save_results_after_each_test=True
    )
    
    # Runner dla MNIST
    xgb_test_runner = XGBTestRunner(
        train_dataset_path='Data/mnist_train.csv',
        test_dataset_path='Data/mnist_test.csv',
        train_data_type=DataType.MNIST_FORMAT,
        test_data_type=DataType.MNIST_FORMAT,
        config=test_runner_config
    )
    
    # Uproszczona przestrzeń - tylko 3 najważniejsze parametry
    space = [
        Real(0.05, 0.2, name='learning_rate'),
        Integer(50, 150, name='n_estimators'),
        Integer(4, 8, name='max_depth'),
    ]
    
    best_accuracy = 0.0
    best_params = None
    iteration = [0]
    
    @use_named_args(space)
    def objective(**params):
        nonlocal best_accuracy, best_params
        iteration[0] += 1
        
        config = XGBTestConfig(
            learning_rate=params['learning_rate'],
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            min_child_weight=1.0,
            gamma=0.0,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            reg_alpha=0.0,
            num_segments=7,
            pixel_normalization_rate=0.34,
            training_set_limit=999999,
            flood_config=FloodConfig(True, True, True, True),
            class_count=10
        )
        
        print(f"\n[{iteration[0]}/10] LR={params['learning_rate']:.3f}, "
              f"N_est={params['n_estimators']}, Depth={params['max_depth']}")
        
        results = xgb_test_runner.run_tests([config])
        accuracy = results[0].accuracy if results else 0.0
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params.copy()
            print(f"  ✨ NEW BEST: {accuracy:.4f}")
        else:
            print(f"  📊 Accuracy: {accuracy:.4f} (Best: {best_accuracy:.4f})")
        
        return -accuracy
    
    # Uruchom optymalizację (tylko 10 iteracji dla testu)
    print("\n🔍 Starting optimization (10 iterations)...\n")
    result = gp_minimize(
        objective,
        space,
        n_calls=10,
        n_random_starts=3,
        random_state=42,
        verbose=False
    )
    
    print("\n" + "="*60)
    print("✅ QUICK TEST COMPLETED!")
    print("="*60)
    print(f"🏆 Best Accuracy: {best_accuracy:.4f}")
    print(f"   Learning rate: {best_params['learning_rate']:.4f}")
    print(f"   N estimators: {best_params['n_estimators']}")
    print(f"   Max depth: {best_params['max_depth']}")
    print("="*60)

if __name__ == "__main__":
    quick_test()
