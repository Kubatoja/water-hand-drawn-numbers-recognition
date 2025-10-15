import csv
from pathlib import Path
from typing import Any, List, Optional
import numpy as np

# Import BFS tylko gdy jest dostępny (warunkowy import)
try:
    from BFS.bfs import calculate_flooded_vector
    BFS_AVAILABLE = True
except ImportError:
    BFS_AVAILABLE = False
    print("Warning: BFS/numba not available, FLOOD_FILL will not work")

from Testers.Shared.models import RawNumberData, VectorNumberData
from Testers.Shared.configs import DimensionalityReductionAlgorithm


class VectorManager:
    """
    Unified class for managing training vectors - handles generation, caching,
    loading from/saving to CSV files, and validation
    """

    def __init__(self, default_vectors_file: str = "Data/vectors.csv"):
        """Initialize VectorManager"""
        self.default_vectors_file = default_vectors_file
        self._cached_vectors: Optional[List[VectorNumberData]] = None
        self._last_config: Optional[Any] = None

    def get_training_vectors(
        self, 
        raw_data: List[RawNumberData],
        config: Any,
        force_regenerate: bool = False,
        auto_save: bool = True
    ) -> List[VectorNumberData]:
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
    def create_vector_for_single_sample(
        raw_number_data: RawNumberData, 
        config: Any
    ) -> VectorNumberData:
        """
        Create a feature vector for a single data sample

        Args:
            raw_number_data: Preprocessed image data
            config: Test configuration containing parameters

        Returns:
            VectorNumberData
        """
        # Obsłuż zarówno FloodConfig obiekt jak i string
        if isinstance(config.flood_config, str):
            flood_str = config.flood_config
        else:
            flood_str = config.flood_config.to_string()
            
        flooded_vector = calculate_flooded_vector(
            raw_number_data.pixels,
            num_segments=config.num_segments,
            floodSides=flood_str
        )
        return VectorNumberData(label=raw_number_data.label, vector=flooded_vector)

    def generate_vectors(
        self, 
        raw_number_data_list: List[RawNumberData], 
        config: Any
    ) -> List[VectorNumberData]:
        """
        Generate vectors for training data using ultra-optimized sequential processing
        
        Args:
            raw_number_data_list: List of raw image data
            config: Test configuration containing parameters
        """
        import time
        
        limit = min(len(raw_number_data_list), config.training_set_limit)
        print(f"Generating {limit} training vectors...")
        
        start_time = time.perf_counter()
        vectors = self._prepare_vectors_batch(raw_number_data_list[:limit], config)
        
        generation_time = time.perf_counter() - start_time
        per_image_ms = generation_time/limit*1000
        throughput = limit/generation_time
        
        return vectors
    
    def _prepare_vectors_batch(
        self, 
        raw_data_list: List[RawNumberData], 
        config: Any
    ) -> List[VectorNumberData]:
        """
        Przygotowuje wektory dla batcha danych (treningowych lub testowych) używając wspólnej logiki.
        """
        import time
        
        # Get image size from config (default to 28 if not present)
        image_size = getattr(config, 'image_size', 28)
        
        # Batch data preparation
        print(f"Batch processing {len(raw_data_list)} samples...")
        all_pixels = np.array([data.pixels for data in raw_data_list])
        all_labels = [data.label for data in raw_data_list]
        
        # Walidacja rozmiaru danych
        expected_pixels = image_size * image_size
        actual_pixels = all_pixels.shape[1] if len(all_pixels.shape) > 1 else all_pixels.shape[0]
        
        if actual_pixels != expected_pixels:
            # Próbuj wykryć rzeczywisty rozmiar
            detected_size = int(np.sqrt(actual_pixels))
            error_msg = (
                f"\n❌ IMAGE SIZE MISMATCH:\n"
                f"   Config expects: {image_size}x{image_size} = {expected_pixels} pixels\n"
                f"   Data contains:  {actual_pixels} pixels"
            )
            if detected_size * detected_size == actual_pixels:
                error_msg += f" ({detected_size}x{detected_size})"
            error_msg += (
                f"\n   → Check dataset configuration: image_size must match actual data!\n"
                f"   → USPS uses 16x16, MNIST/EMNIST use 28x28"
            )
            raise ValueError(error_msg)
        
        binarized_batch = np.where(all_pixels > config.pixel_normalization_rate, 1, 0).reshape(-1, image_size, image_size)
        original_batch = all_pixels.reshape(-1, image_size, image_size)
        
        # Utwórz podstawowe wektory - zawsze binarne, bo apply_dimensionality_reduction obsłuży resztę
        vectors = [VectorNumberData(label=label, vector=binarized_batch[i].flatten().astype(float).tolist()) for i, label in enumerate(all_labels)]
        
        # Zastosuj redukcję wymiarów
        if config.dimensionality_reduction_algorithm != DimensionalityReductionAlgorithm.NONE:
            print(f"Applying {config.dimensionality_reduction_algorithm.value} dimensionality reduction...")
            vectors = self.apply_dimensionality_reduction(vectors, binarized_batch, original_batch, config)
        
        return vectors
    
    def apply_dimensionality_reduction(
        self, 
        vectors: List[VectorNumberData], 
        binarized_batch: np.ndarray,
        original_batch: np.ndarray,
        config: Any
    ) -> List[VectorNumberData]:
        """
        Applies dimensionality reduction to the vectors if configured.
        
        Args:
            vectors: List of VectorNumberData
            config: Test configuration containing reduction parameters
            
        Returns:
            List of VectorNumberData with reduced dimensions
        """
        if config.dimensionality_reduction_algorithm == DimensionalityReductionAlgorithm.NONE:
            return vectors
            
        print(f"Applying {config.dimensionality_reduction_algorithm.value} dimensionality reduction...")
        
        # Extract features and labels
        X = np.array([v.vector for v in vectors])
        y = np.array([v.label for v in vectors])
        
        # Apply reduction
        if config.dimensionality_reduction_algorithm == DimensionalityReductionAlgorithm.FLOOD_FILL:
            if not BFS_AVAILABLE:
                raise ImportError("BFS/numba not available. Cannot use FLOOD_FILL dimensionality reduction.")

            # Obsłuż zarówno FloodConfig obiekt jak i string
            if isinstance(config.flood_config, str):
                flood_str = config.flood_config
            else:
                flood_str = config.flood_config.to_string()

            reduced_vectors = []
            for i, vector_data in enumerate(vectors):
                flooded_vector = calculate_flooded_vector(
                    binarized_batch[i],
                    num_segments=config.num_segments,
                    floodSides=flood_str
                )
                reduced_vectors.append(VectorNumberData(label=vector_data.label, vector=flooded_vector))
            return reduced_vectors

        else:
            # For statistical methods (PCA, LDA, Isomap, t-SNE) and NONE: use original images
            X = original_batch.reshape(original_batch.shape[0], -1)  # Flatten to 2D
            y = np.array([v.label for v in vectors])

            if config.dimensionality_reduction_algorithm == DimensionalityReductionAlgorithm.NONE:
                # Return flattened original images
                reduced_vectors = [
                    VectorNumberData(label=label, vector=X[i].tolist())
                    for i, label in enumerate(y)
                ]
                return reduced_vectors

            elif config.dimensionality_reduction_algorithm == DimensionalityReductionAlgorithm.PCA:
                from sklearn.decomposition import PCA
                reducer = PCA(n_components=config.dimensionality_reduction_n_components)
                X_reduced = reducer.fit_transform(X)

            elif config.dimensionality_reduction_algorithm == DimensionalityReductionAlgorithm.LDA:
                from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
                reducer = LinearDiscriminantAnalysis(n_components=config.dimensionality_reduction_n_components)
                X_reduced = reducer.fit_transform(X, y)

            elif config.dimensionality_reduction_algorithm == DimensionalityReductionAlgorithm.ISOMAP:
                from sklearn.manifold import Isomap
                reducer = Isomap(n_components=config.dimensionality_reduction_n_components)
                X_reduced = reducer.fit_transform(X)

            elif config.dimensionality_reduction_algorithm == DimensionalityReductionAlgorithm.TSNE:
                from sklearn.manifold import TSNE
                reducer = TSNE(n_components=config.dimensionality_reduction_n_components, random_state=42)
                X_reduced = reducer.fit_transform(X)

            else:
                raise ValueError(f"Unsupported dimensionality reduction algorithm: {config.dimensionality_reduction_algorithm}")

            # Create new vectors
            reduced_vectors = [
                VectorNumberData(label=label, vector=X_reduced[i].tolist())
                for i, label in enumerate(y)
            ]

            print(f"Reduced dimensions from {X.shape[1]} to {X_reduced.shape[1]}")
            return reduced_vectors

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

            first_row = next(reader, None)
            if first_row and not first_row[0].isdigit():
                pass
            else:
                file.seek(0)
                reader = csv.reader(file)

            for row in reader:
                if not row:
                    continue

                try:
                    label = int(row[0])
                    vector = [float(x) for x in row[1:]]
                    vectors.append(VectorNumberData(label=label, vector=vector))
                except (ValueError, IndexError) as e:
                    print(f"Warning: Skipping invalid row: {row}. Error: {e}")

        print(f"Loaded {len(vectors)} vectors from {file_path}")
        return vectors

    def save_vectors_to_csv(
        self, 
        vectors: List[VectorNumberData],
        output_file: str = None,
        include_header: bool = True
    ) -> None:
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

            if include_header:
                vector_size = len(vectors[0].vector)
                header = ['label'] + [f'feature_{i}' for i in range(vector_size)]
                writer.writerow(header)

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

        expected_size = len(vectors[0].vector)
        for i, vector in enumerate(vectors):
            if len(vector.vector) != expected_size:
                print(f"Warning: Vector {i} has size {len(vector.vector)}, "
                      f"expected {expected_size}")
                return False

        labels = [v.label for v in vectors]
        unique_labels = set(labels)
        print(f"Found labels: {sorted(unique_labels)}")

        return True
    
    def _should_regenerate(self, new_config: Any) -> bool:
        """Sprawdza czy należy regenerować wektory"""
        if self._last_config is None:
            return True
            
        old_config = self._last_config
        
        if hasattr(old_config, 'pixel_normalization_rate') and hasattr(new_config, 'pixel_normalization_rate'):
            if old_config.pixel_normalization_rate != new_config.pixel_normalization_rate:
                return True
                
        if hasattr(old_config, 'flood_config') and hasattr(new_config, 'flood_config'):
            if old_config.flood_config != new_config.flood_config:
                return True
                
        if hasattr(old_config, 'training_set_limit') and hasattr(new_config, 'training_set_limit'):
            if old_config.training_set_limit != new_config.training_set_limit:
                return True
                
        if hasattr(old_config, 'num_segments') and hasattr(new_config, 'num_segments'):
            if old_config.num_segments != new_config.num_segments:
                return True
                
        if hasattr(old_config, 'dimensionality_reduction_algorithm') and hasattr(new_config, 'dimensionality_reduction_algorithm'):
            if old_config.dimensionality_reduction_algorithm != new_config.dimensionality_reduction_algorithm:
                return True
                
        if hasattr(old_config, 'dimensionality_reduction_n_components') and hasattr(new_config, 'dimensionality_reduction_n_components'):
            if old_config.dimensionality_reduction_n_components != new_config.dimensionality_reduction_n_components:
                return True
                
        return False
