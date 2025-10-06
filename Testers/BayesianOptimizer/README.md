# Bayesian Optimizer for XGBoost

Clean, maintainable, production-ready implementation of Bayesian Optimization for XGBoost hyperparameter tuning.

## 🎯 Features

- ✅ **SOLID principles** - Single Responsibility, Dependency Injection
- ✅ **Clean Code** - Readable, maintainable, well-documented
- ✅ **Type hints** - Full type annotations for better IDE support
- ✅ **Modular design** - Easy to extend and customize
- ✅ **CLI interface** - Simple command-line usage
- ✅ **Comprehensive examples** - Multiple usage patterns documented
- ✅ **Production ready** - Error handling, logging, result saving

## 📦 Installation

```bash
pip install scikit-optimize
```

**Important:** Package name is `scikit-optimize` (with hyphen), imported as `skopt`.

## 🚀 Quick Start

```bash
# Quick test (10 iterations)
python main_bayesian_optimized.py --quick

# Full optimization (50 iterations)
python main_bayesian_optimized.py

# MNIST only
python main_bayesian_optimized.py --mnist

# Custom iterations
python main_bayesian_optimized.py --iterations 100
```

## 📖 Usage

### Simple usage

```python
from Testers.BayesianOptimizer import (
    OptimizationOrchestrator,
    MNIST_DATASET,
    FULL_SEARCH_SPACE
)

orchestrator = OptimizationOrchestrator(
    datasets=[MNIST_DATASET],
    search_space_config=FULL_SEARCH_SPACE,
    n_iterations=50
)

results = orchestrator.run_optimization()
```

### Custom search space

```python
from Testers.BayesianOptimizer import SearchSpaceConfig

custom_space = SearchSpaceConfig(
    learning_rate_min=0.05,
    learning_rate_max=0.15,
    max_depth_min=5,
    max_depth_max=8
)

orchestrator = OptimizationOrchestrator(
    datasets=[MNIST_DATASET],
    search_space_config=custom_space,
    n_iterations=30
)
```

More examples in `Testers/BayesianOptimizer/examples.py`

## 🏗️ Architecture

```
Testers/BayesianOptimizer/
├── BayesianOptimizer.py    # Core optimization logic
├── orchestrator.py          # Multi-dataset coordination
├── configs.py               # Search space configuration
├── dataset_config.py        # Dataset definitions
├── reporter.py              # Result formatting
├── examples.py              # Usage examples
└── API_DOCUMENTATION.md     # Full API docs
```

### Key Components

- **BayesianOptimizer**: Core optimization algorithm
- **OptimizationOrchestrator**: Coordinates optimization for multiple datasets
- **SearchSpaceConfig**: Defines hyperparameter ranges
- **DatasetConfig**: Dataset configuration
- **ResultReporter**: Formats and displays results

## 🎨 Design Principles

### Single Responsibility Principle

Each class has one clear responsibility:
- `BayesianOptimizer` → Optimization logic only
- `Orchestrator` → Coordination only
- `Reporter` → Reporting only
- `Configs` → Configuration only

### Dependency Injection

```python
optimizer = BayesianOptimizer(
    test_runner=test_runner,    # Injected
    search_space=search_space,  # Injected
    fixed_params=fixed_params   # Injected
)
```

### Type Safety

```python
def optimize(self) -> OptimizationResult:
    """Full type hints for better IDE support and type checking."""
```

### Clean Code

- Descriptive names
- Small, focused methods
- No magic numbers
- Comprehensive docstrings
- DRY (Don't Repeat Yourself)

## 📊 Optimized Hyperparameters

- `learning_rate` (0.01 - 0.3)
- `n_estimators` (50 - 300)
- `max_depth` (3 - 10)
- `min_child_weight` (1.0 - 5.0)
- `gamma` (0.0 - 0.5)
- `subsample` (0.6 - 1.0)
- `colsample_bytree` (0.6 - 1.0)
- `reg_lambda` (0.1 - 2.0)
- `reg_alpha` (0.0 - 1.0)

## 🧪 Testing

```python
# Run examples
python Testers/BayesianOptimizer/examples.py

# Quick test
python main_bayesian_optimized.py --quick
```

## 📚 Documentation

- **Quick Start**: `QUICK_START.md`
- **API Documentation**: `Testers/BayesianOptimizer/API_DOCUMENTATION.md`
- **Examples**: `Testers/BayesianOptimizer/examples.py`
- **Theory**: `BAYESIAN_OPTIMIZATION_README.md`

## 🔧 Extending

### Add new hyperparameter

```python
# In SearchSpaceConfig
my_param_min: float = 0.0
my_param_max: float = 1.0

# In to_search_space()
Real(self.my_param_min, self.my_param_max, name='my_param')
```

### Add new dataset

```python
from Testers.BayesianOptimizer import DatasetConfig

MY_DATASET = DatasetConfig(
    name='My Dataset',
    train_path='Data/my_train.csv',
    test_path='Data/my_test.csv',
    data_type=DataType.MNIST_FORMAT,
    class_count=10
)
```

## 🆚 Comparison

| Approach | Tests | Time | Intelligence | Code Quality |
|----------|-------|------|--------------|--------------|
| Grid Search (old) | 1000+ | ~40h | ❌ None | ⚠️ Monolithic |
| **Bayesian (new)** | **50** | **~5h** | ✅ **High** | ✅ **Clean** |

## 🎯 Benefits of Refactoring

### Before (main_bayesian.py)
- ❌ All logic in one file
- ❌ Hard to test
- ❌ Difficult to extend
- ❌ Poor separation of concerns

### After (BayesianOptimizer package)
- ✅ Modular architecture
- ✅ Easy to test
- ✅ Simple to extend
- ✅ Clear responsibilities
- ✅ Reusable components
- ✅ Better documentation

## 📝 License

Part of water-hand-drawn-numbers-recognition project.

## 👨‍💻 Development

Built following:
- SOLID principles
- Clean Code practices
- PEP 8 style guide
- Type hints (PEP 484)
- Docstring conventions (PEP 257)

## 🤝 Contributing

When adding features:
1. Follow existing architecture
2. Maintain single responsibility
3. Add type hints
4. Document with docstrings
5. Add examples if needed
6. Update API documentation

---

**Happy Optimizing! 🚀**
