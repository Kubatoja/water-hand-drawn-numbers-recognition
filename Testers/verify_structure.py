"""
Szybki test weryfikacyjny struktury XGBoost Testera
"""

def test_imports():
    """Test czy wszystkie importy działają poprawnie"""
    print("Testing imports...")
    
    try:
        # Shared components
        from Testers.Shared.models import TestResult, RawNumberData, VectorNumberData
        from Testers.Shared.configs import FloodConfig, TestRunnerConfig
        from Testers.Shared.DataLoader import DataLoader, DataType
        from Testers.Shared.VectorManager import VectorManager
        from Testers.Shared.MetricsCalculator import MetricsCalculator
        from Testers.Shared.ResultSaver import ResultsSaver
        from Testers.Shared.TestResultCollector import TestResultCollector
        print("✅ Shared components imported successfully")
        
        # XGBoost components
        from Testers.XgBoostTester.configs import XGBTestConfig, XGBTestConfigField, FieldConfig
        from Testers.XgBoostTester.XGBTester import XGBTester
        from Testers.XgBoostTester.XGBTestRunner import XGBTestRunner
        from Testers.XgBoostTester.TestConfigFactory import create_xgb_test_configs
        print("✅ XGBoost components imported successfully")
        
        # ANN components (updated)
        from Testers.AnnTester.configs import ANNTestConfig, ANNTestConfigField
        from Testers.AnnTester.ANNTester import ANNTester
        from Testers.AnnTester.ANNTestRunner import ANNTestRunner
        print("✅ ANN components imported successfully")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_creation():
    """Test tworzenia konfiguracji"""
    print("\nTesting config creation...")
    
    try:
        from Testers.XgBoostTester.configs import XGBTestConfig
        from Testers.Shared.configs import FloodConfig
        
        config = XGBTestConfig(
            learning_rate=0.1,
            n_estimators=100,
            max_depth=6,
            min_child_weight=1.0,
            gamma=0.0,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            reg_alpha=0.0,
            num_segments=4,
            pixel_normalization_rate=0.5,
            training_set_limit=1000,
            flood_config=FloodConfig.from_string("1111"),
            class_count=10
        )
        
        print(f"✅ XGBTestConfig created successfully")
        print(f"   Learning rate: {config.learning_rate}")
        print(f"   N estimators: {config.n_estimators}")
        print(f"   Max depth: {config.max_depth}")
        
        return True
    except Exception as e:
        print(f"❌ Config creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_factory():
    """Test fabryki konfiguracji"""
    print("\nTesting config factory...")
    
    try:
        from Testers.XgBoostTester.configs import XGBTestConfig, XGBTestConfigField, FieldConfig
        from Testers.XgBoostTester.TestConfigFactory import create_xgb_test_configs
        from Testers.Shared.configs import FloodConfig
        
        default_config = XGBTestConfig(
            learning_rate=0.1,
            n_estimators=100,
            max_depth=6,
            min_child_weight=1.0,
            gamma=0.0,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            reg_alpha=0.0,
            num_segments=4,
            pixel_normalization_rate=0.5,
            training_set_limit=1000,
            flood_config=FloodConfig.from_string("1111"),
            class_count=10
        )
        
        field_configs = [
            FieldConfig(
                field_name=XGBTestConfigField.LEARNING_RATE,
                start=0.05,
                stop=0.15,
                step=0.05
            )
        ]
        
        configs = create_xgb_test_configs(
            field_configs=field_configs,
            generate_combinations=False,
            default_config=default_config
        )
        
        print(f"✅ Config factory works - generated {len(configs)} configs")
        for i, cfg in enumerate(configs):
            print(f"   Config {i+1}: learning_rate={cfg.learning_rate}")
        
        return True
    except Exception as e:
        print(f"❌ Config factory failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metrics_calculator():
    """Test kalkulatora metryk"""
    print("\nTesting metrics calculator...")
    
    try:
        from Testers.Shared.MetricsCalculator import MetricsCalculator
        import numpy as np
        
        calc = MetricsCalculator()
        
        # Przykładowe dane
        actual = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        predicted = np.array([0, 1, 2, 0, 2, 1, 0, 1, 2])
        
        metrics = calc.calculate_all_metrics(actual, predicted, num_classes=3)
        
        print(f"✅ Metrics calculator works")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall: {metrics['recall']:.4f}")
        print(f"   F1-Score: {metrics['f1_score']:.4f}")
        
        return True
    except Exception as e:
        print(f"❌ Metrics calculator failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Uruchom wszystkie testy"""
    print("=" * 60)
    print("XGBoost Tester - Verification Tests")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Config Creation Test", test_config_creation),
        ("Config Factory Test", test_config_factory),
        ("Metrics Calculator Test", test_metrics_calculator)
    ]
    
    results = []
    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    all_passed = all(result for _, result in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All tests PASSED! Structure is ready to use.")
    else:
        print("⚠️  Some tests FAILED. Please review errors above.")
    print("=" * 60)


if __name__ == "__main__":
    main()
