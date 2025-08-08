from pathlib import Path
from typing import Optional, List

from Tester.ResultSaver import ResultsSaver

class TestResultCollector:
    """Zbiera i zarządza wynikami testów"""

    def __init__(self):
        self.results: List['TestResult'] = []
        self.failed_tests: List[tuple] = []
        self.saver = ResultsSaver()  # Single instance for incremental saves

    def add_success(self, result: 'TestResult'):
        """
        Dodaje udany wynik testu

        Args:
            result: Wynik testu
        """
        self.results.append(result)

    def add_success_and_save(self, result: 'TestResult', test_index: int):
        """
        Dodaje wynik testu i od razu go zapisuje (dla incremental saves)

        Args:
            result: Wynik testu
            test_index: Indeks testu
        """
        self.results.append(result)
        self.saver.save_single_result(result, test_index)

    def add_failure(self, test_index: int, error: str):
        """Dodaje nieudany test"""
        self.failed_tests.append((test_index, error))

    def print_summary(self, total_tests: int):
        """Wyświetla podsumowanie"""
        print("=" * 60)
        print(f"Test suite completed!")
        print(f"Successful tests: {len(self.results)}/{total_tests}")

        if self.failed_tests:
            print(f"Failed tests: {len(self.failed_tests)}")
            for index, error in self.failed_tests:
                print(f"  Test #{index + 1}: {error}")

    def save_results(self, custom_path: str = None):
        """Zapisuje wszystkie wyniki do pliku (final save) - POPRAWIONA WERSJA"""
        if self.results:
            # If we haven't been saving incrementally, use the existing saver instance
            # If we have been saving incrementally, just finalize
            if not self.saver._initialized:
                # First time save - initialize and save all
                self.saver.save(self.results, custom_path)
            else:
                # We've been saving incrementally, results are already saved
                print(f"Results already saved incrementally to: {self.saver.get_output_directory()}")
        else:
            print("No results to save")

    def create_final_report(self) -> Path:
        """Tworzy końcowy raport"""
        return self.saver.create_final_summary_report()

    def get_results_directory(self) -> Optional[Path]:
        """Zwraca katalog z wynikami"""
        return self.saver.get_output_directory()