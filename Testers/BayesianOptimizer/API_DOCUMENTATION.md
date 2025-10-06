# Bayesian Optimizer API Documentation

## Architektura

```
Testers/BayesianOptimizer/
├── __init__.py                 # Public API
├── BayesianOptimizer.py        # Core optimizer logic
├── orchestrator.py             # Multi-dataset orchestration
├── configs.py                  # Search space configuration
├── dataset_config.py           # Dataset definitions
├── reporter.py                 # Result reporting
└── examples.py                 # Usage examples
```

## Kluczowe Komponenty

### 1. BayesianOptimizer
**Odpowiedzialność:** Implementacja algorytmu Bayesian Optimization

**Użycie:**
```python
from Testers.BayesianOptimizer import BayesianOptimizer

optimizer = BayesianOptimizer(
    test_runner=my_test_runner,
    search_space=search_space,
    fixed_params=fixed_params,
    n_iterations=50,
    n_random_starts=10
)

result = optimizer.optimize()
```

**Parametry:**
- `test_runner` (XGBTestRunner): Runner do wykonywania testów
- `search_space` (List[Dimension]): Przestrzeń przeszukiwania z skopt
- `fixed_params` (Dict): Parametry stałe (np. class_count)
- `n_iterations` (int): Liczba iteracji optymalizacji
- `n_random_starts` (int): Liczba początkowych losowych prób
- `random_state` (int): Seed dla reprodukowalności
- `verbose` (bool): Czy wyświetlać logi

**Zwraca:** `OptimizationResult`

---

### 2. OptimizationOrchestrator
**Odpowiedzialność:** Koordynacja optymalizacji dla wielu datasetów

**Użycie:**
```python
from Testers.BayesianOptimizer import OptimizationOrchestrator, ALL_DATASETS, FULL_SEARCH_SPACE

orchestrator = OptimizationOrchestrator(
    datasets=ALL_DATASETS,
    search_space_config=FULL_SEARCH_SPACE,
    n_iterations=50
)

results = orchestrator.run_optimization()
```

**Parametry:**
- `datasets` (List[DatasetConfig]): Lista datasetów do testowania
- `search_space_config` (SearchSpaceConfig): Konfiguracja przestrzeni
- `n_iterations` (int): Liczba iteracji na dataset
- `n_random_starts` (int): Liczba losowych startów
- `test_runner_config` (TestRunnerConfig): Opcjonalna konfiguracja runnera
- `verbose` (bool): Czy wyświetlać logi

**Zwraca:** `Dict[str, OptimizationResult]` - wyniki per dataset

---

### 3. SearchSpaceConfig
**Odpowiedzialność:** Definicja zakresów hiperparametrów

**Użycie:**
```python
from Testers.BayesianOptimizer import SearchSpaceConfig

# Własna przestrzeń
custom_space = SearchSpaceConfig(
    learning_rate_min=0.05,
    learning_rate_max=0.15,
    max_depth_min=5,
    max_depth_max=8
)

# Konwersja do formatu skopt
search_space = custom_space.to_search_space()
```

**Predefiniowane:**
- `FULL_SEARCH_SPACE` - Pełna przestrzeń (wszystkie zakresy)
- `QUICK_SEARCH_SPACE` - Zawężona przestrzeń (szybsze testy)

**Parametry (wszystkie opcjonalne):**
- `learning_rate_min/max` (float): Zakres learning rate [0.01, 0.3]
- `n_estimators_min/max` (int): Zakres liczby drzew [50, 300]
- `max_depth_min/max` (int): Zakres głębokości [3, 10]
- `min_child_weight_min/max` (float): [1.0, 5.0]
- `gamma_min/max` (float): [0.0, 0.5]
- `subsample_min/max` (float): [0.6, 1.0]
- `colsample_bytree_min/max` (float): [0.6, 1.0]
- `reg_lambda_min/max` (float): [0.1, 2.0]
- `reg_alpha_min/max` (float): [0.0, 1.0]

---

### 4. DatasetConfig
**Odpowiedzialność:** Definicja datasetu do testowania

**Użycie:**
```python
from Testers.BayesianOptimizer import DatasetConfig
from Testers.Shared.DataLoader import DataType

my_dataset = DatasetConfig(
    name='Custom Dataset',
    train_path='Data/train.csv',
    test_path='Data/test.csv',
    data_type=DataType.MNIST_FORMAT,
    class_count=10
)
```

**Predefiniowane:**
- `MNIST_DATASET` - MNIST (10 klas)
- `EMNIST_DATASET` - EMNIST-Balanced (47 klas)
- `ALL_DATASETS` - Lista obu powyższych

---

### 5. FixedParamsConfig
**Odpowiedzialność:** Parametry nie podlegające optymalizacji

**Użycie:**
```python
from Testers.BayesianOptimizer import FixedParamsConfig

fixed = FixedParamsConfig(
    num_segments=7,
    pixel_normalization_rate=0.34,
    class_count=10
)

params_dict = fixed.to_dict()
```

**Parametry:**
- `num_segments` (int): Liczba segmentów [domyślnie: 7]
- `pixel_normalization_rate` (float): Próg normalizacji [domyślnie: 0.34]
- `training_set_limit` (int): Limit danych treningowych [domyślnie: 999999]
- `flood_config` (FloodConfig): Konfiguracja flood fill [domyślnie: wszystkie True]
- `class_count` (int): Liczba klas [domyślnie: 10]

---

### 6. OptimizationResult
**Odpowiedzialność:** Przechowywanie wyników optymalizacji

**Pola:**
- `best_accuracy` (float): Najlepsza osiągnięta accuracy
- `best_params` (Dict): Parametry najlepszej konfiguracji
- `all_iterations` (List[Dict]): Historia wszystkich iteracji
- `dataset_name` (str): Nazwa datasetu

**Metody:**
- `top_results(n: int)` - Zwraca top N wyników

**Użycie:**
```python
result = optimizer.optimize()

print(f"Best: {result.best_accuracy:.4f}")
print(f"Params: {result.best_params}")

# Top 5 wyników
for config in result.top_results(5):
    print(f"Accuracy: {config['accuracy']:.4f}")
```

---

### 7. ResultReporter
**Odpowiedzialność:** Formatowanie i wyświetlanie wyników

**Metody:**
```python
from Testers.BayesianOptimizer import ResultReporter

reporter = ResultReporter()

# Podsumowanie datasetu
reporter.print_dataset_summary(result)

# Finalne podsumowanie wszystkich
reporter.print_final_summary(results_dict)
```

---

## Przepływ Danych

```
main_bayesian_optimized.py
    ↓
OptimizationOrchestrator
    ↓ (dla każdego datasetu)
    ├── Tworzy XGBTestRunner
    ├── Tworzy SearchSpace (z SearchSpaceConfig)
    ├── Tworzy FixedParams (z FixedParamsConfig)
    ↓
BayesianOptimizer
    ↓ (n_iterations razy)
    ├── gp_minimize wybiera parametry
    ├── Tworzy XGBTestConfig
    ├── Uruchamia test (XGBTestRunner)
    ├── Zbiera TestResult
    ↓
OptimizationResult
    ↓
ResultReporter
    ↓
Console output + Saved results
```

---

## Wzorce Projektowe

### 1. **Single Responsibility Principle (SRP)**
- `BayesianOptimizer` - tylko optymalizacja
- `Orchestrator` - tylko koordynacja
- `Reporter` - tylko raportowanie
- `Configs` - tylko konfiguracja

### 2. **Dependency Injection**
```python
optimizer = BayesianOptimizer(
    test_runner=test_runner,  # Injected dependency
    search_space=search_space,  # Injected configuration
    ...
)
```

### 3. **Factory Pattern**
```python
# SearchSpaceConfig działa jako factory
search_space = config.to_search_space()
```

### 4. **Builder Pattern**
```python
# SearchSpaceConfig jako builder
config = SearchSpaceConfig(
    learning_rate_min=0.05,
    learning_rate_max=0.15
)
```

### 5. **Strategy Pattern**
- Różne `SearchSpaceConfig` = różne strategie przeszukiwania

---

## Testowanie

### Unit Tests
```python
# test_bayesian_optimizer.py
def test_optimizer_initialization():
    optimizer = BayesianOptimizer(...)
    assert optimizer.n_iterations == 50

def test_search_space_conversion():
    config = SearchSpaceConfig()
    space = config.to_search_space()
    assert len(space) == 9
```

### Integration Tests
```python
# test_integration.py
def test_full_optimization_flow():
    orchestrator = OptimizationOrchestrator(...)
    results = orchestrator.run_optimization()
    assert 'MNIST' in results
```

---

## Rozszerzanie

### Dodanie nowego hiperparametru

1. **W `SearchSpaceConfig`:**
```python
my_param_min: float = 0.0
my_param_max: float = 1.0
```

2. **W `to_search_space()`:**
```python
Real(self.my_param_min, self.my_param_max, name='my_param')
```

3. Parametr automatycznie trafi do `XGBTestConfig`

### Dodanie nowego datasetu

```python
from Testers.BayesianOptimizer import DatasetConfig

NEW_DATASET = DatasetConfig(
    name='New Dataset',
    train_path='Data/new_train.csv',
    test_path='Data/new_test.csv',
    data_type=DataType.MNIST_FORMAT,
    class_count=20
)
```

---

## Best Practices

1. **Zawsze używaj Orchestratora** dla standardowych przypadków
2. **Bezpośredni BayesianOptimizer** tylko dla advanced use cases
3. **Predefiniowane konfiguracje** (FULL_SEARCH_SPACE) zamiast custom
4. **Verbose=True** podczas developmentu, False w produkcji
5. **Zapisuj wyniki** (automatyczne przez TestRunnerConfig)

---

## CLI Interface

```bash
# Pełna optymalizacja
python main_bayesian_optimized.py

# Szybki test
python main_bayesian_optimized.py --quick

# Tylko MNIST
python main_bayesian_optimized.py --mnist

# Custom liczba iteracji
python main_bayesian_optimized.py --iterations 100
```
