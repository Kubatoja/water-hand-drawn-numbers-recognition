"""
Bayesian Optimizer package for XGBoost hyperparameter optimization.
"""
from Testers.BayesianOptimizer.BayesianOptimizer import BayesianOptimizer, OptimizationResult
from Testers.BayesianOptimizer.orchestrator import OptimizationOrchestrator
from Testers.BayesianOptimizer.configs import (
    SearchSpaceConfig,
    FixedParamsConfig,
    QUICK_SEARCH_SPACE,
    FULL_SEARCH_SPACE,
    MNIST_FIXED_PARAMS,
    EMNIST_FIXED_PARAMS
)
from Testers.Shared.dataset_config import (
    DatasetConfig,
    # Podstawowe datasety
    MNIST_DATASET,
    FASHION_MNIST_DATASET,
    EMNIST_BALANCED_DATASET,
    EMNIST_DIGITS_DATASET,
    ARABIC_DATASET,
    USPS_DATASET,
    # MNIST-C
    MNIST_C_IDENTITY,
    MNIST_C_BRIGHTNESS,
    MNIST_C_CANNY_EDGES,
    MNIST_C_DOTTED_LINE,
    MNIST_C_FOG,
    MNIST_C_GLASS_BLUR,
    MNIST_C_IMPULSE_NOISE,
    MNIST_C_MOTION_BLUR,
    MNIST_C_ROTATE,
    MNIST_C_SCALE,
    MNIST_C_SHEAR,
    MNIST_C_SHOT_NOISE,
    MNIST_C_SPATTER,
    MNIST_C_STRIPE,
    MNIST_C_TRANSLATE,
    MNIST_C_ZIGZAG,
    # Kolekcje
    ALL_DATASETS,
    BASIC_DATASETS,
    DIGITS_ONLY_DATASETS,
    NON_DIGITS_DATASETS,
    ALL_EMNIST_DATASETS,
    ALL_MNIST_C_DATASETS,
    # Helper functions
    get_dataset,
    get_datasets_by_group,
    get_datasets_by_names,
    create_mnist_c_config,
)
from Testers.BayesianOptimizer.reporter import ResultReporter
from Testers.Shared.dataset_enums import DatasetName, MnistCCorruption, DatasetGroup

__all__ = [
    # Core
    'BayesianOptimizer',
    'OptimizationResult',
    'OptimizationOrchestrator',
    'ResultReporter',
    
    # Configs
    'SearchSpaceConfig',
    'FixedParamsConfig',
    'DatasetConfig',
    'QUICK_SEARCH_SPACE',
    'FULL_SEARCH_SPACE',
    'MNIST_FIXED_PARAMS',
    'EMNIST_FIXED_PARAMS',
    
    # Enums
    'DatasetName',
    'MnistCCorruption',
    'DatasetGroup',
    
    # Podstawowe datasety
    'MNIST_DATASET',
    'FASHION_MNIST_DATASET',
    'EMNIST_BALANCED_DATASET',
    'EMNIST_DIGITS_DATASET',
    'ARABIC_DATASET',
    'USPS_DATASET',
    
    # MNIST-C
    'MNIST_C_IDENTITY',
    'MNIST_C_BRIGHTNESS',
    'MNIST_C_CANNY_EDGES',
    'MNIST_C_DOTTED_LINE',
    'MNIST_C_FOG',
    'MNIST_C_GLASS_BLUR',
    'MNIST_C_IMPULSE_NOISE',
    'MNIST_C_MOTION_BLUR',
    'MNIST_C_ROTATE',
    'MNIST_C_SCALE',
    'MNIST_C_SHEAR',
    'MNIST_C_SHOT_NOISE',
    'MNIST_C_SPATTER',
    'MNIST_C_STRIPE',
    'MNIST_C_TRANSLATE',
    'MNIST_C_ZIGZAG',
    
    # Kolekcje
    'ALL_DATASETS',
    'BASIC_DATASETS',
    'DIGITS_ONLY_DATASETS',
    'NON_DIGITS_DATASETS',
    'ALL_EMNIST_DATASETS',
    'ALL_MNIST_C_DATASETS',
    
    # Helpers
    'get_dataset',
    'get_datasets_by_group',
    'get_datasets_by_names',
    'create_mnist_c_config',
]

