"""
Konfiguracja przestrzeni przeszukiwania dla Bayesian Optimization.
Odpowiedzialność: Definicja zakresów hiperparametrów.
"""
from typing import List, Dict, Any
from dataclasses import dataclass
from skopt.space import Real, Integer, Dimension

from Testers.Shared.configs import FloodConfig


@dataclass
class SearchSpaceConfig:
    """Konfiguracja przestrzeni przeszukiwania."""
    
    # Vectorization parameters (NOWE - teraz optymalizowane!)
    num_segments_min: int = 3
    num_segments_max: int = 10
    
    pixel_normalization_rate_min: float = 0.1
    pixel_normalization_rate_max: float = 0.5
    
    # XGBoost hyperparameters ranges
    learning_rate_min: float = 0.01
    learning_rate_max: float = 0.3
    
    n_estimators_min: int = 50
    n_estimators_max: int = 300
    
    max_depth_min: int = 3
    max_depth_max: int = 10
    
    min_child_weight_min: float = 1.0
    min_child_weight_max: float = 5.0
    
    gamma_min: float = 0.0
    gamma_max: float = 0.5
    
    subsample_min: float = 0.6
    subsample_max: float = 1.0
    
    colsample_bytree_min: float = 0.6
    colsample_bytree_max: float = 1.0
    
    reg_lambda_min: float = 0.1
    reg_lambda_max: float = 2.0
    
    reg_alpha_min: float = 0.0
    reg_alpha_max: float = 1.0
    
    def to_search_space(self) -> List[Dimension]:
        """Konwertuje konfigurację do formatu skopt."""
        return [
            # Vectorization parameters (NOWE!)
            Integer(self.num_segments_min, self.num_segments_max, name='num_segments'),
            Real(self.pixel_normalization_rate_min, self.pixel_normalization_rate_max, name='pixel_normalization_rate'),
            
            # XGBoost parameters
            Real(self.learning_rate_min, self.learning_rate_max, name='learning_rate'),
            Integer(self.n_estimators_min, self.n_estimators_max, name='n_estimators'),
            Integer(self.max_depth_min, self.max_depth_max, name='max_depth'),
            Real(self.min_child_weight_min, self.min_child_weight_max, name='min_child_weight'),
            Real(self.gamma_min, self.gamma_max, name='gamma'),
            Real(self.subsample_min, self.subsample_max, name='subsample'),
            Real(self.colsample_bytree_min, self.colsample_bytree_max, name='colsample_bytree'),
            Real(self.reg_lambda_min, self.reg_lambda_max, name='reg_lambda'),
            Real(self.reg_alpha_min, self.reg_alpha_max, name='reg_alpha'),
        ]


@dataclass
class FixedParamsConfig:
    """Parametry stałe (nie optymalizowane)."""
    
    # UWAGA: num_segments i pixel_normalization_rate są teraz OPTYMALIZOWANE
    # i znajdują się w SearchSpaceConfig!
    
    training_set_limit: int = 999999
    flood_config: FloodConfig = None
    class_count: int = 10
    image_size: int = 28  # Rozmiar obrazu (domyślnie 28x28)
    
    def __post_init__(self):
        if self.flood_config is None:
            self.flood_config = FloodConfig(True, True, True, True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Konwertuje do słownika."""
        return {
            'training_set_limit': self.training_set_limit,
            'flood_config': self.flood_config,
            'class_count': self.class_count,
            'image_size': self.image_size
        }


# Predefiniowane konfiguracje
QUICK_SEARCH_SPACE = SearchSpaceConfig(
    # Vectorization (zredukowane dla quick mode)
    num_segments_min=5,
    num_segments_max=8,
    pixel_normalization_rate_min=0.2,
    pixel_normalization_rate_max=0.4,
    
    # XGBoost
    learning_rate_min=0.05,
    learning_rate_max=0.2,
    n_estimators_min=50,
    n_estimators_max=150,
    max_depth_min=4,
    max_depth_max=8,
)

FULL_SEARCH_SPACE = SearchSpaceConfig()  # Używa domyślnych wartości (pełny zakres)

MNIST_FIXED_PARAMS = FixedParamsConfig(class_count=10, image_size=28)
EMNIST_FIXED_PARAMS = FixedParamsConfig(class_count=47, image_size=28)
