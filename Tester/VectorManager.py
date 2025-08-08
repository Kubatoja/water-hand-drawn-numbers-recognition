import csv
from pathlib import Path
from typing import List, Optional

from BFS.bfs import calculate_flooded_vector
from Tester.configs import ANNTestConfig
from Tester.otherModels import RawNumberData, VectorNumberData


class VectorManager:
    """
    Unified class for managing training vectors - handles generation, caching,
    loading from/saving to CSV files, and validation
    """

    def __init__(self, default_vectors_file: str = "Data/vectors.csv"):
        """Initialize VectorManager"""
        self.default_vectors_file = default_vectors_file
        self._cached_vectors: Optional[List['VectorNumberData']] = None
        self._last_config: Optional['ANNTestConfig'] = None

    def get_training_vectors(self, raw_data: List['RawNumberData'],
                             config: 'ANNTestConfig',
                             force_regenerate: bool = False,
                             auto_save: bool = True) -> List['VectorNumberData']:
        """Returns training vectors, regenerates only if needed"""

        if (self._cached_vectors is None or
                force_regenerate or
                self._should_regenerate(config)):

            print("Generating new vectors...")
            self._cached_vectors = self.generate_vectors(raw_data, config)
            self._last_config = config

            if auto_save:
                self.save_vectors_to_csv(self._cached_vectors)

        return self._cached_vectors

    @staticmethod
    def create_vector_for_single_sample(rawNumberData: RawNumberData, config: ANNTestConfig) -> VectorNumberData:
        """
        Create a feature vector for a single data sample

        Args:
            rawNumberData: Preprocessed image data
            config: Test configuration containing parameters

        Returns:
            VectorNumberData
        """
        flooded_vector = calculate_flooded_vector(
            rawNumberData.pixels,
            num_segments=config.num_segments,
            floodSides=config.flood_config.to_string()
        )
        return VectorNumberData(label=rawNumberData.label, vector=flooded_vector)

    def generate_vectors(self, rawNumberDataList: List[RawNumberData], config: ANNTestConfig) -> List[VectorNumberData]:
        """
        Generate vectors for training data and save to file

        Args:
            rawNumberDataList: List
            config: Test configuration containing parameters
        """

        print(f"Generating {max(len(rawNumberDataList), config.training_set_limit)} training vectors...")
        vectors = []
        for i in range(config.training_set_limit):
            if i % 1000 == 0:  # Progress indicator
                print(f"Processing sample {i}/{config.training_set_limit}")

            raw_number_data = rawNumberDataList[i]
            raw_number_data.binarize_data(config.pixel_normalization_rate)
            vector = self.create_vector_for_single_sample(raw_number_data, config)
            vectors.append(vector)
        return vectors

    def load_vectors_from_csv(self, input_file: str = None) -> List[VectorNumberData]:
        """
        Wczytuje VectorNumberData z pliku CSV

        Args:
            input_file: Ścieżka do pliku wejściowego, jeśli None używa domyślnej

        Returns:
            Lista obiektów VectorNumberData
        """
        file_path = input_file or self.default_vectors_file

        if not Path(file_path).exists():
            raise FileNotFoundError(f"Vectors file not found: {file_path}")

        vectors = []
        with open(file_path, 'r') as file:
            reader = csv.reader(file)

            # Sprawdź czy pierwszy wiersz to nagłówek
            first_row = next(reader, None)
            if first_row and not first_row[0].isdigit():
                # Pomiń nagłówek
                pass
            else:
                # Pierwszy wiersz to dane, cofnij pointer
                file.seek(0)
                reader = csv.reader(file)

            for row in reader:
                if not row:  # Pomiń puste wiersze
                    continue

                try:
                    label = int(row[0])
                    vector = [float(x) for x in row[1:]]
                    vectors.append(VectorNumberData(label=label, vector=vector))
                except (ValueError, IndexError) as e:
                    print(f"Warning: Skipping invalid row: {row}. Error: {e}")

        print(f"Loaded {len(vectors)} vectors from {file_path}")
        return vectors

    def save_vectors_to_csv(self, vectors: List[VectorNumberData],
                            output_file: str = None,
                            include_header: bool = True) -> None:
        """
        Zapisuje VectorNumberData do pliku CSV

        Args:
            vectors: Lista obiektów VectorNumberData do zapisania
            output_file: Ścieżka do pliku wyjściowego, jeśli None używa domyślnej
            include_header: Czy dołączyć nagłówek
        """
        if not vectors:
            print("Warning: No vectors to save")
            return

        file_path = Path(output_file or self.default_vectors_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)

            # Zapisz nagłówek
            if include_header:
                vector_size = len(vectors[0].vector)
                header = ['label'] + [f'feature_{i}' for i in range(vector_size)]
                writer.writerow(header)

            # Zapisz dane
            for vector_data in vectors:
                row = [vector_data.label] + list(vector_data.vector)
                writer.writerow(row)

        print(f"Saved {len(vectors)} vectors to {file_path}")

    def validate_vectors(self, vectors: List[VectorNumberData]) -> bool:
        """
        Waliduje spójność danych wektorowych

        Args:
            vectors: Lista wektorów do walidacji

        Returns:
            True jeśli dane są spójne, False w przeciwnym razie
        """
        if not vectors:
            return True

        # Sprawdź czy wszystkie wektory mają ten sam rozmiar
        expected_size = len(vectors[0].vector)
        for i, vector in enumerate(vectors):
            if len(vector.vector) != expected_size:
                print(f"Warning: Vector {i} has size {len(vector.vector)}, "
                      f"expected {expected_size}")
                return False

        # Sprawdź czy etykiety są w dozwolonym zakresie
        labels = [v.label for v in vectors]
        unique_labels = set(labels)
        print(f"Found labels: {sorted(unique_labels)}")

        return True
    def _should_regenerate(oldConfig: ANNTestConfig, newConfig: ANNTestConfig) -> bool:
        if oldConfig.pixel_normalization_rate != newConfig.pixel_normalization_rate:
            return True
        if oldConfig.flood_config != newConfig.flood_config:
            return True
        if oldConfig.training_set_limit != newConfig.training_set_limit:
            return True
        if oldConfig.num_segments != newConfig.num_segments:
            return True
        return False
