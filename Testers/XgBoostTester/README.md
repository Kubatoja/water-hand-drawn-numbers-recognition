# XGBoost Tester - Przykłady użycia

## Podstawowe użycie

```python
from Testers.XgBoostTester.XGBTestRunner import XGBTestRunner
from Testers.XgBoostTester.configs import XGBTestConfig
from Testers.Shared.configs import FloodConfig, TestRunnerConfig
from Testers.Shared.DataLoader import DataType

# Konfiguracja podstawowa
config = XGBTestConfig(
    # Parametry XGBoost
    learning_rate=0.1,
    n_estimators=100,
    max_depth=6,
    min_child_weight=1.0,
    gamma=0.0,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    reg_alpha=0.0,
    
    # Parametry wektorów
    num_segments=4,
    pixel_normalization_rate=0.5,
    training_set_limit=1000,
    flood_config=FloodConfig.from_string("1111"),
    
    # Dataset info
    class_count=10
)

# Runner config
runner_config = TestRunnerConfig(
    skip_first_vector_generation=False,
    save_results_after_each_test=True
)

# Uruchomienie testów
runner = XGBTestRunner(
    train_dataset_path="Data/mnist_train.csv",
    test_dataset_path="Data/mnist_test.csv",
    train_data_type=DataType.MNIST_FORMAT,
    test_data_type=DataType.MNIST_FORMAT,
    config=runner_config,
    vectors_file="Data/vectors_xgb.csv"
)

results = runner.run_tests([config])
```

## Generowanie wielu konfiguracji

```python
from Testers.XgBoostTester.TestConfigFactory import create_xgb_test_configs
from Testers.XgBoostTester.configs import XGBTestConfig, XGBTestConfigField, FieldConfig
from Testers.Shared.configs import FloodConfig

# Konfiguracja domyślna
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

# Definiuj zakresy parametrów do testowania
field_configs = [
    FieldConfig(
        field_name=XGBTestConfigField.LEARNING_RATE,
        start=0.01,
        stop=0.3,
        step=0.05
    ),
    FieldConfig(
        field_name=XGBTestConfigField.MAX_DEPTH,
        start=3,
        stop=10,
        step=2
    ),
    FieldConfig(
        field_name=XGBTestConfigField.N_ESTIMATORS,
        start=50,
        stop=200,
        step=50
    )
]

# Generuj konfiguracje (jeden parametr na raz)
test_configs = create_xgb_test_configs(
    field_configs=field_configs,
    generate_combinations=False,  # True dla wszystkich kombinacji
    default_config=default_config
)

# Uruchom testy
runner = XGBTestRunner(
    train_dataset_path="Data/mnist_train.csv",
    test_dataset_path="Data/mnist_test.csv"
)

results = runner.run_tests(test_configs)
```

## Analiza wyników

Wyniki są automatycznie zapisywane w folderze `results/test_results_XGBoost_TIMESTAMP/`:
- `test_results_summary.csv` - podsumowanie wszystkich testów
- `confusion_matrix_test_X.csv` - confusion matrix dla każdego testu
- `final_summary_report.txt` - raport końcowy

### Metryki w wynikach:
- **Accuracy** - ogólna dokładność
- **Precision** - precyzja (macro-average)
- **Recall** - recall (macro-average)
- **F1-Score** - F1 score (macro-average)
- **Per-class metrics** - metryki dla każdej klasy osobno

## Porównanie z ANN Tester

```python
from Testers.AnnTester.ANNTestRunner import ANNTestRunner
from Testers.AnnTester.configs import ANNTestConfig
from Testers.XgBoostTester.XGBTestRunner import XGBTestRunner
from Testers.XgBoostTester.configs import XGBTestConfig
from Testers.Shared.configs import FloodConfig

# Wspólne parametry
common_params = {
    "num_segments": 4,
    "pixel_normalization_rate": 0.5,
    "training_set_limit": 5000,
    "flood_config": FloodConfig.from_string("1111"),
    "class_count": 10
}

# ANN Config
ann_config = ANNTestConfig(
    trees_count=10,
    leaves_count=50,
    **common_params
)

# XGBoost Config
xgb_config = XGBTestConfig(
    learning_rate=0.1,
    n_estimators=100,
    max_depth=6,
    min_child_weight=1.0,
    gamma=0.0,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    reg_alpha=0.0,
    **common_params
)

# Uruchom oba testy
ann_runner = ANNTestRunner("Data/mnist_train.csv", "Data/mnist_test.csv")
xgb_runner = XGBTestRunner("Data/mnist_train.csv", "Data/mnist_test.csv")

ann_results = ann_runner.run_tests([ann_config])
xgb_results = xgb_runner.run_tests([xgb_config])

# Porównaj wyniki
print(f"ANN Accuracy: {ann_results[0].accuracy:.4f}")
print(f"XGB Accuracy: {xgb_results[0].accuracy:.4f}")
```

## Struktura folderów

```
Testers/
├── Shared/                    # Komponenty wspólne
│   ├── configs.py            # FloodConfig, TestRunnerConfig
│   ├── models.py             # TestResult, RawNumberData, VectorNumberData
│   ├── DataLoader.py         # Wczytywanie danych
│   ├── VectorManager.py      # Zarządzanie wektorami
│   ├── MetricsCalculator.py  # Obliczanie metryk
│   ├── ResultSaver.py        # Zapisywanie wyników
│   └── TestResultCollector.py # Zbieranie wyników
│
├── AnnTester/                # Tester dla ANN
│   ├── configs.py            # ANNTestConfig
│   ├── ANNTester.py          # Logika testowania
│   ├── ANNTestRunner.py      # Runner
│   └── TestConfigFactory.py  # Generator konfiguracji
│
└── XgBoostTester/            # Tester dla XGBoost
    ├── configs.py            # XGBTestConfig
    ├── XGBTester.py          # Logika testowania
    ├── XGBTestRunner.py      # Runner
    └── TestConfigFactory.py  # Generator konfiguracji
```
