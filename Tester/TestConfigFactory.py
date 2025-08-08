# Enum dla nazw pól
import copy
import itertools
from dataclasses import asdict
from typing import List

from Tester.ANNTester import ANNTestConfig
from Tester.configs import FieldConfig




def create_ann_test_configs(
        field_configs: List[FieldConfig],
        generate_combinations: bool,
        default_config: ANNTestConfig
) -> List[ANNTestConfig]:
    """
    Tworzy listę obiektów ANNTestConfig na podstawie konfiguracji pól.

    Args:
        field_configs: Lista obiektów FieldConfig zdefiniowanych dla każdego pola.
        generate_combinations: Jeśli True, generuje wszystkie kombinacje. Jeśli False,
                               zmienia tylko jeden parametr na raz.
        default_config: Obiekt ANNTestConfig z domyślnymi wartościami.

    Returns:
        Lista obiektów ANNTestConfig zawierająca testy.
    """

    # Przygotowanie wartości dla każdego pola
    field_values = {}
    for config in field_configs:
        if isinstance(config.start, (int, float)):
            values = list(range(int(config.start), int(config.stop) + 1, int(config.step)))
        else:
            values = [config.start, config.stop]
        field_values[config.field_name.value] = values

    result_configs = []

    if generate_combinations:
        # Generowanie wszystkich kombinacji
        keys = list(field_values.keys())
        combinations = list(itertools.product(*field_values.values()))

        for combo in combinations:
            cfg = copy.deepcopy(default_config)  # zachowanie typów zagnieżdżonych
            for i, key in enumerate(keys):
                setattr(cfg, key, combo[i])
            result_configs.append(cfg)

    else:
        # Generowanie tylko jednego zmienionego parametru na raz
        for key, values in field_values.items():
            for value in values:
                cfg = copy.deepcopy(default_config)  # zachowanie typów zagnieżdżonych
                setattr(cfg, key, value)
                result_configs.append(cfg)

    return result_configs