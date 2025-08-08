import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd

class ResultsSaver:
    """Klasa odpowiedzialna za eksport wyników testów do plików CSV."""

    def __init__(self):
        """Initialize ResultsSaver with tracking for incremental saves"""
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
        # Convert single result to list for uniform processing
        if not isinstance(test_results, list):
            test_results = [test_results]

        if not test_results:
            raise ValueError("Lista wyników testów nie może być pusta")

        # Initialize output directory if not done yet
        if not self._initialized or output_dir:
            self._initialize_output_directory(output_dir)

        try:
            # Generate/update summary CSV with all tests
            self._generate_or_update_summary_csv(test_results)

            # Generate confusion matrix files for new tests
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
        # Initialize output directory if not done yet
        if not self._initialized:
            self._initialize_output_directory()

        try:
            # Append single result to summary CSV
            self._append_single_result_to_summary(test_result, test_index)

            # Generate confusion matrix file for this test
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
            self._output_dir = Path(f"results/test_results_{timestamp}")

        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._summary_file = self._output_dir / "test_results_summary.csv"

        # Create header if file doesn't exist
        if not self._summary_file.exists():
            self._create_summary_header()

        self._initialized = True

    def _create_summary_header(self) -> None:
        """Tworzy nagłówek w pliku summary CSV"""
        fieldnames = self._get_fieldnames()

        with open(self._summary_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    def _get_fieldnames(self) -> List[str]:
        """Zwraca listę nazw kolumn dla pliku CSV"""
        return [
            'test_id',
            'training_time',
            'train_set_size',
            'test_set_size',
            'execution_time',
            'correct_predictions',
            'incorrect_predictions',
            'accuracy',
            'trees_count',
            'leaves_count',
            'num_segments',
            'pixel_normalization_rate',
            'training_set_limit'
        ]

    def _generate_or_update_summary_csv(self, test_results: List['TestResult']) -> None:
        """Generuje lub aktualizuje główny plik CSV z wynikami"""
        fieldnames = self._get_fieldnames()

        # Read existing data to determine next test_id
        existing_test_count = self._get_existing_test_count()

        with open(self._summary_file, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            for idx, result in enumerate(test_results):
                test_id = existing_test_count + idx + 1
                row_data = self._create_row_data(result, test_id)
                writer.writerow(row_data)

    def _append_single_result_to_summary(self, test_result: 'TestResult', test_index: int) -> None:
        """Dodaje pojedynczy wynik do pliku summary"""
        fieldnames = self._get_fieldnames()

        with open(self._summary_file, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            row_data = self._create_row_data(test_result, test_index + 1)
            writer.writerow(row_data)

    def _create_row_data(self, result: 'TestResult', test_id: int) -> dict:
        """Tworzy dane wiersza dla wyniku testu"""
        return {
            'test_id': test_id,
            'training_time': result.training_time,
            'train_set_size': result.train_set_size,
            'test_set_size': result.test_set_size,
            'execution_time': result.execution_time,
            'correct_predictions': result.correct_predictions,
            'incorrect_predictions': result.incorrect_predictions,
            'accuracy': result.accuracy,
            'trees_count': result.config.trees_count,
            'leaves_count': result.config.leaves_count,
            'num_segments': result.config.num_segments,
            'pixel_normalization_rate': result.config.pixel_normalization_rate,
            'training_set_limit': result.config.training_set_limit
        }

    def _get_existing_test_count(self) -> int:
        """Zwraca liczbę istniejących testów w pliku summary"""
        if not self._summary_file.exists():
            return 0

        try:
            with open(self._summary_file, 'r', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                # Skip header and count rows
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

        # Konwersja numpy array do DataFrame
        df = pd.DataFrame(
            result.confusion_matrix,
            index=[f"Actual_{i}" for i in range(result.confusion_matrix.shape[0])],
            columns=[f"Predicted_{i}" for i in range(result.confusion_matrix.shape[1])]
        )

        # Zapis do pliku CSV
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

        # Read summary data
        df = pd.read_csv(self._summary_file)

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("RAPORT KOŃCOWY TESTÓW ANN\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Liczba przeprowadzonych testów: {len(df)}\n")
            f.write(f"Data wygenerowania: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Basic statistics
            f.write("STATYSTYKI PODSTAWOWE:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Średnia dokładność: {df['accuracy'].mean():.4f}\n")
            f.write(f"Najwyższa dokładność: {df['accuracy'].max():.4f}\n")
            f.write(f"Najniższa dokładność: {df['accuracy'].min():.4f}\n")
            f.write(f"Odchylenie standardowe: {df['accuracy'].std():.4f}\n\n")

            # Best performing test
            best_test = df.loc[df['accuracy'].idxmax()]
            f.write("NAJLEPSZY WYNIK:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Test ID: {best_test['test_id']}\n")
            f.write(f"Dokładność: {best_test['accuracy']:.4f}\n")
            f.write(f"Liczba drzew: {best_test['trees_count']}\n")
            f.write(f"Liczba liści: {best_test['leaves_count']}\n")
            f.write(f"Segmenty: {best_test['num_segments']}\n")

        print(f"Raport końcowy zapisany w: {report_file}")
        return report_file

