import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from Testers.Shared.models import TestResult


def convert_numpy_types_to_native(obj: Any) -> Any:
    """
    Konwertuje numpy typy na natywne typy Pythona (dla JSON/CSV).
    Rekurencyjnie przetwarza dict i list.
    
    Args:
        obj: Obiekt do konwersji (może być numpy type, dict, list, etc.)
        
    Returns:
        Obiekt z przekonwertowanymi typami
        
    Examples:
        >>> convert_numpy_types_to_native(np.int64(42))
        42
        >>> convert_numpy_types_to_native({'a': np.float64(3.14)})
        {'a': 3.14}
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types_to_native(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types_to_native(item) for item in obj]
    return obj


class ResultsSaver:
    """Klasa odpowiedzialna za eksport wyników testów do plików CSV."""

    def __init__(self, algorithm_name: str = "model"):
        """
        Initialize ResultsSaver
        
        Args:
            algorithm_name: Nazwa algorytmu (np. 'ANN', 'XGBoost') dla nazewnictwa
        """
        self.algorithm_name = algorithm_name
        self._output_dir: Optional[Path] = None
        self._summary_file: Optional[Path] = None
        self._initialized = False

    def save(self, test_results: Union[List['TestResult'], 'TestResult'],
             output_dir: Optional[str] = None) -> Path:
        """
        Eksportuje wyniki testów do plików CSV.

        Args:
            test_results: Lista wyników testów lub pojedynczy wynik
            output_dir: Opcjonalna ścieżka do folderu (dla custom path)

        Returns:
            Path: Ścieżka do utworzonego folderu z plikami

        Raises:
            ValueError: Gdy wyniki są puste
            OSError: Gdy wystąpi problem z utworzeniem folderu lub pliku
        """
        if not isinstance(test_results, list):
            test_results = [test_results]

        if not test_results:
            raise ValueError("Lista wyników testów nie może być pusta")

        if not self._initialized or output_dir:
            self._initialize_output_directory(output_dir)

        try:
            self._generate_or_update_summary_csv(test_results)
            self._generate_confusion_matrix_files(test_results)

            print(f"Pomyślnie wyeksportowano {len(test_results)} wyników testów do: {self._output_dir}")
            return self._output_dir

        except OSError as e:
            raise OSError(f"Błąd podczas tworzenia plików: {e}") from e

    def save_single_result(self, test_result: 'TestResult', test_index: int) -> Path:
        """
        Zapisuje pojedynczy wynik testu, dodając go do istniejącego pliku summary.

        Args:
            test_result: Wynik pojedynczego testu
            test_index: Indeks testu (dla numeracji)

        Returns:
            Path: Ścieżka do folderu z plikami
        """
        if not self._initialized:
            self._initialize_output_directory()

        try:
            self._append_single_result_to_summary(test_result, test_index)
            self._generate_single_confusion_matrix_file(test_result, test_index)

            print(f"Dodano wynik testu #{test_index + 1} do pliku summary")
            return self._output_dir

        except OSError as e:
            raise OSError(f"Błąd podczas zapisywania pojedynczego wyniku: {e}") from e

    def _initialize_output_directory(self, custom_path: Optional[str] = None) -> None:
        """Inicjalizuje katalog wyjściowy i główny plik CSV"""
        if custom_path:
            self._output_dir = Path(custom_path)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._output_dir = Path(f"results/test_results_{self.algorithm_name}_{timestamp}")

        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._summary_file = self._output_dir / "test_results_summary.csv"

        if not self._summary_file.exists():
            self._create_summary_header()

        self._initialized = True

    def _create_summary_header(self) -> None:
        """Tworzy nagłówek w pliku summary CSV"""
        fieldnames = self._get_base_fieldnames()

        with open(self._summary_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    def _get_base_fieldnames(self) -> List[str]:
        """Zwraca bazowe nazwy kolumn dla pliku CSV"""
        return [
            'test_id',
            'training_time',
            'train_set_size',
            'test_set_size',
            'execution_time',
            'correct_predictions',
            'incorrect_predictions',
            'accuracy',
            'precision',
            'recall',
            'f1_score',
            'pixel_normalization_rate',
            'num_segments',
            'training_set_limit'
        ]

    def _extract_config_params(self, config: Any) -> Dict[str, Any]:
        """
        Ekstraktuje parametry z konfiguracji do zapisu w CSV
        
        Args:
            config: Obiekt konfiguracji (ANNTestConfig lub XGBTestConfig)
            
        Returns:
            Słownik z parametrami
        """
        params = {}
        
        # Parametry wspólne
        if hasattr(config, 'pixel_normalization_rate'):
            params['pixel_normalization_rate'] = config.pixel_normalization_rate
        if hasattr(config, 'num_segments'):
            params['num_segments'] = config.num_segments
        if hasattr(config, 'training_set_limit'):
            params['training_set_limit'] = config.training_set_limit
        
        # Dataset name - zawsze jako jeden z pierwszych
        if hasattr(config, 'dataset_name'):
            params['dataset_name'] = config.dataset_name
            
        # Parametry specyficzne dla algorytmu - dynamiczne dodawanie
        for attr in dir(config):
            if not attr.startswith('_') and attr not in [
                'pixel_normalization_rate', 'num_segments', 
                'training_set_limit', 'class_count', 'flood_config', 'dataset_name'
            ]:
                value = getattr(config, attr, None)
                if value is not None and not callable(value):
                    params[attr] = value
                    
        return params

    def _generate_or_update_summary_csv(self, test_results: List['TestResult']) -> None:
        """Generuje lub aktualizuje główny plik CSV z wynikami"""
        existing_test_count = self._get_existing_test_count()

        # Zbierz wszystkie możliwe nazwy kolumn
        all_fieldnames = set(self._get_base_fieldnames())
        for result in test_results:
            config_params = self._extract_config_params(result.config)
            all_fieldnames.update(config_params.keys())
        
        fieldnames = sorted(list(all_fieldnames))
        # Upewnij się, że test_id jest pierwszy
        if 'test_id' in fieldnames:
            fieldnames.remove('test_id')
        fieldnames = ['test_id'] + fieldnames

        # Jeśli plik już istnieje, sprawdź czy potrzebujemy dodać kolumny
        if self._summary_file.exists():
            existing_df = pd.read_csv(self._summary_file)
            new_columns = set(fieldnames) - set(existing_df.columns)
            if new_columns:
                # Dodaj nowe kolumny
                for col in new_columns:
                    existing_df[col] = None
                existing_df = existing_df[fieldnames]  # Uporządkuj kolumny
                existing_df.to_csv(self._summary_file, index=False)

        with open(self._summary_file, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            for idx, result in enumerate(test_results):
                test_id = existing_test_count + idx + 1
                row_data = self._create_row_data(result, test_id, fieldnames)
                writer.writerow(row_data)

    def _append_single_result_to_summary(self, test_result: 'TestResult', test_index: int) -> None:
        """Dodaje pojedynczy wynik do pliku summary"""
        # Read existing to get all possible fieldnames
        all_fieldnames = set(self._get_base_fieldnames())
        config_params = self._extract_config_params(test_result.config)
        all_fieldnames.update(config_params.keys())
        
        fieldnames = sorted(list(all_fieldnames))
        if 'test_id' in fieldnames:
            fieldnames.remove('test_id')
        fieldnames = ['test_id'] + fieldnames

        # Update file structure if needed
        if self._summary_file.exists():
            existing_df = pd.read_csv(self._summary_file)
            new_columns = set(fieldnames) - set(existing_df.columns)
            if new_columns:
                for col in new_columns:
                    existing_df[col] = None
                existing_df = existing_df[fieldnames]
                existing_df.to_csv(self._summary_file, index=False)

        with open(self._summary_file, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            row_data = self._create_row_data(test_result, test_index + 1, fieldnames)
            writer.writerow(row_data)

    def _create_row_data(self, result: 'TestResult', test_id: int, 
                        fieldnames: List[str]) -> Dict[str, Any]:
        """Tworzy dane wiersza dla wyniku testu"""
        row = {
            'test_id': test_id,
            'training_time': result.training_time,
            'train_set_size': result.train_set_size,
            'test_set_size': result.test_set_size,
            'execution_time': result.execution_time,
            'correct_predictions': result.correct_predictions,
            'incorrect_predictions': result.incorrect_predictions,
            'accuracy': result.accuracy,
            'precision': result.precision,
            'recall': result.recall,
            'f1_score': result.f1_score,
        }
        
        # Dodaj parametry konfiguracji
        config_params = self._extract_config_params(result.config)
        row.update(config_params)
        
        # Wypełnij brakujące kolumny None
        for field in fieldnames:
            if field not in row:
                row[field] = None
                
        return row

    def _get_existing_test_count(self) -> int:
        """Zwraca liczbę istniejących testów w pliku summary"""
        if not self._summary_file.exists():
            return 0

        try:
            with open(self._summary_file, 'r', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                return sum(1 for _ in reader) - 1
        except Exception:
            return 0

    def _generate_confusion_matrix_files(self, test_results: List['TestResult']) -> None:
        """Generuje pliki confusion matrix dla listy testów"""
        existing_test_count = self._get_existing_test_count() - len(test_results)

        for idx, result in enumerate(test_results):
            test_id = existing_test_count + idx + 1
            self._generate_single_confusion_matrix_file(result, test_id - 1)

    def _generate_single_confusion_matrix_file(self, result: 'TestResult', test_index: int) -> None:
        """Generuje plik confusion matrix dla pojedynczego testu"""
        matrix_file = self._output_dir / f"confusion_matrix_test_{test_index + 1}.csv"

        df = pd.DataFrame(
            result.confusion_matrix,
            index=[f"Actual_{i}" for i in range(result.confusion_matrix.shape[0])],
            columns=[f"Predicted_{i}" for i in range(result.confusion_matrix.shape[1])]
        )

        df.to_csv(matrix_file, encoding='utf-8')

    def get_output_directory(self) -> Optional[Path]:
        """Zwraca aktualny katalog wyjściowy"""
        return self._output_dir

    def get_summary_file_path(self) -> Optional[Path]:
        """Zwraca ścieżkę do pliku summary"""
        return self._summary_file

    def get_test_count(self) -> int:
        """Zwraca liczbę zapisanych testów"""
        return self._get_existing_test_count()

    def reset(self) -> None:
        """Resetuje stan SaveResulta dla nowej sesji"""
        self._output_dir = None
        self._summary_file = None
        self._initialized = False

    def create_final_summary_report(self) -> Path:
        """
        Tworzy końcowy raport podsumowujący wszystkie testy

        Returns:
            Path: Ścieżka do pliku raportu
        """
        if not self._summary_file or not self._summary_file.exists():
            raise ValueError("Brak danych do utworzenia raportu")

        report_file = self._output_dir / "final_summary_report.txt"
        df = pd.read_csv(self._summary_file)

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"RAPORT KOŃCOWY TESTÓW {self.algorithm_name.upper()}\n")
            f.write("=" * 50 + "\n\n")

            # Informacja o datasecie jeśli dostępna
            if 'dataset_name' in df.columns and pd.notna(df['dataset_name'].iloc[0]):
                dataset_name = df['dataset_name'].iloc[0]
                f.write(f"Dataset: {dataset_name}\n")
            
            f.write(f"Liczba przeprowadzonych testów: {len(df)}\n")
            f.write(f"Data wygenerowania: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("STATYSTYKI PODSTAWOWE:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Średnia dokładność: {df['accuracy'].mean():.4f}\n")
            f.write(f"Najwyższa dokładność: {df['accuracy'].max():.4f}\n")
            f.write(f"Najniższa dokładność: {df['accuracy'].min():.4f}\n")
            f.write(f"Odchylenie standardowe: {df['accuracy'].std():.4f}\n\n")
            
            # Dodatkowe metryki jeśli dostępne
            if 'precision' in df.columns:
                f.write(f"Średnia precyzja: {df['precision'].mean():.4f}\n")
            if 'recall' in df.columns:
                f.write(f"Średni recall: {df['recall'].mean():.4f}\n")
            if 'f1_score' in df.columns:
                f.write(f"Średni F1-score: {df['f1_score'].mean():.4f}\n\n")

            best_test = df.loc[df['accuracy'].idxmax()]
            f.write("NAJLEPSZY WYNIK:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Test ID: {best_test['test_id']}\n")
            f.write(f"Dokładność: {best_test['accuracy']:.4f}\n")
            
            # Wypisz wszystkie parametry konfiguracji
            config_columns = [col for col in df.columns if col not in [
                'test_id', 'training_time', 'train_set_size', 'test_set_size',
                'execution_time', 'correct_predictions', 'incorrect_predictions',
                'accuracy', 'precision', 'recall', 'f1_score'
            ]]
            
            f.write("\nParametry konfiguracji:\n")
            for col in config_columns:
                if pd.notna(best_test[col]):
                    f.write(f"  {col}: {best_test[col]}\n")

        print(f"Raport końcowy zapisany w: {report_file}")
        return report_file
