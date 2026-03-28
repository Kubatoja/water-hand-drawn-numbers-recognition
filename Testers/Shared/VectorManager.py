import json
import time
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
        self._cached_test_vectors: Optional[List[VectorNumberData]] = None
        self._last_test_config_key: Optional[dict] = None
        # Przechowuje dopasowany reduktor (np. PCA, LDA) oraz jego konfigurację
        # dzięki temu można użyć tego samego reduktora dla testowych wektorów
        # zamiast dopasowywać osobny reduktor na zbiorze testowym.
        self._last_reducer = None
        self._last_reducer_config = None

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

            # Sprawdź czy możemy wczytać z pliku zamiast generować
            if not force_regenerate and self._can_load_from_file(config):
                try:
                    print("Loading vectors from file...")
                    self._cached_vectors = self.load_vectors_from_csv()
                    self._last_config = config
                    self._cached_test_vectors = None
                    return self._cached_vectors
                except (FileNotFoundError, ValueError) as e:
                    print(f"Could not load vectors from file ({e}), generating new ones...")

            print("Generating new vectors...")
            self._cached_vectors = self.generate_vectors(raw_data, config)
            self._last_config = config
            self._cached_test_vectors = None

            if auto_save:
                self.save_vectors_to_csv(self._cached_vectors)
                self._save_config_metadata(config)

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
        limit = min(len(raw_number_data_list), config.training_set_limit)
        print(f"Generating {limit} training vectors...")
        
        start_time = time.perf_counter()
        vectors = self._prepare_vectors_batch(raw_number_data_list[:limit], config, is_training_data=True)
        
        generation_time = time.perf_counter() - start_time
        if limit > 0:
            per_image_ms = generation_time / limit * 1000
            throughput = limit / generation_time if generation_time > 0 else 0.0
            print(f"Vector generation throughput: {throughput:.2f} samples/s ({per_image_ms:.3f} ms/sample)")
        
        return vectors
    
    def _prepare_vectors_batch(
        self, 
        raw_data_list: List[RawNumberData], 
        config: Any,
        is_training_data: bool = False
    ) -> List[VectorNumberData]:
        """
        Przygotowuje wektory dla batcha danych (treningowych lub testowych) używając wspólnej logiki.
        """
        # Get image size from config (default to 28 if not present)
        image_size = getattr(config, 'image_size', 28)
        
        # Batch data preparation
        print(f"Batch processing {len(raw_data_list)} samples...")
        all_pixels = np.asarray([data.pixels for data in raw_data_list], dtype=np.float64)
        all_labels = np.asarray([data.label for data in raw_data_list], dtype=np.int64)
        
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

        if config.dimensionality_reduction_algorithm == DimensionalityReductionAlgorithm.NONE:
            feature_matrix = binarized_batch.reshape(binarized_batch.shape[0], -1).astype(np.float64, copy=False)
        else:
            feature_matrix = self.apply_dimensionality_reduction(
                binarized_batch,
                original_batch,
                all_labels,
                config,
                is_training_data=is_training_data
            )

        return [
            VectorNumberData(label=int(all_labels[i]), vector=feature_matrix[i].tolist())
            for i in range(feature_matrix.shape[0])
        ]
    
    def apply_dimensionality_reduction(
        self, 
        binarized_batch: np.ndarray,
        original_batch: np.ndarray,
        labels: np.ndarray,
        config: Any,
        is_training_data: bool = False
    ) -> np.ndarray:
        """
        Applies dimensionality reduction to the vectors if configured.
        
        Args:
            vectors: List of VectorNumberData
            config: Test configuration containing reduction parameters
            is_training_data: True if processing training data, False for test data
            
        Returns:
            List of VectorNumberData with reduced dimensions
        """
        if config.dimensionality_reduction_algorithm == DimensionalityReductionAlgorithm.NONE:
            return binarized_batch.reshape(binarized_batch.shape[0], -1).astype(np.float64, copy=False)
            
        print(f"Applying {config.dimensionality_reduction_algorithm.value} dimensionality reduction...")
        y = labels
        
        # Apply reduction
        if config.dimensionality_reduction_algorithm == DimensionalityReductionAlgorithm.FLOOD_FILL:
            if not BFS_AVAILABLE:
                raise ImportError("BFS/numba not available. Cannot use FLOOD_FILL dimensionality reduction.")

            # Obsłuż zarówno FloodConfig obiekt jak i string
            if isinstance(config.flood_config, str):
                flood_str = config.flood_config
            else:
                flood_str = config.flood_config.to_string()

            return np.asarray([
                calculate_flooded_vector(
                    binarized_batch[i],
                    num_segments=config.num_segments,
                    floodSides=flood_str
                )
                for i in range(binarized_batch.shape[0])
            ], dtype=np.float64)

        else:
            # For statistical methods (PCA, LDA, Isomap, UMAP) and NONE: use original images
            X = original_batch.reshape(original_batch.shape[0], -1)  # Flatten to 2D

            # Key opisujący konfigurację redukcji (używany do ponownego użycia reduktora)
            config_key = (config.dimensionality_reduction_algorithm, config.dimensionality_reduction_n_components, config.training_set_limit)

            # Jeśli mamy już dopasowany reduktor o tej samej konfiguracji, użyj transform
            if self._last_reducer is not None and self._last_reducer_config == config_key:
                try:
                    print("Reusing previously fitted reducer for transform...")
                    X_reduced = self._last_reducer.transform(X)
                except Exception as e:
                    if not is_training_data:
                        # KRYTYCZNE: Dla danych testowych transform() MUSI działać
                        # Nie pozwalamy na ponowne dopasowanie reduktora na danych testowych
                        raise ValueError(
                            f"\n❌ KRYTYCZNY BŁĄD: Nie można transformować danych testowych!\n"
                            f"   Transform() zawiódł z błędem: {e}\n"
                            f"   Reduktor musi być prawidłowo dopasowany na danych treningowych.\n"
                            f"   Algorytm: {config.dimensionality_reduction_algorithm.value}\n"
                            f"   Sprawdź czy konfiguracja się nie zmieniła między treningiem a testem."
                        ) from e
                    # Dla danych treningowych można pozwolić na nowe dopasowanie
                    print(f"Warning: Transform failed on training data, will refit. Error: {e}")
                    X_reduced = None
            else:
                X_reduced = None

            # Jeśli nie mamy X_reduced, dopasuj nowy reduktor i zapisz go
            if X_reduced is None:
                if config.dimensionality_reduction_algorithm == DimensionalityReductionAlgorithm.PCA:
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

                elif config.dimensionality_reduction_algorithm == DimensionalityReductionAlgorithm.UMAP:
                    # UMAP oferuje transform() i jest lepiej dopasowany do pipeline train->test.
                    try:
                        import umap
                        reducer = umap.UMAP(n_components=config.dimensionality_reduction_n_components, random_state=42)
                        X_reduced = reducer.fit_transform(X)
                    except Exception:
                        # Jeśli umap nie jest zainstalowany, podnieśmy błąd — UMAP powinien być dostępny.
                        raise

                elif config.dimensionality_reduction_algorithm == DimensionalityReductionAlgorithm.PACMAP:
                    try:
                        import pacmap
                    except ImportError as e:
                        raise ImportError("PaCMAP is required for DimensionalityReductionAlgorithm.PACMAP. Install via pip install pacmap") from e

                    reducer = pacmap.PaCMAP(n_components=config.dimensionality_reduction_n_components, random_state=42)
                    X_reduced = reducer.fit_transform(X)

                elif config.dimensionality_reduction_algorithm == DimensionalityReductionAlgorithm.TRIMAP:
                    try:
                        import trimap
                    except ImportError as e:
                        raise ImportError("TriMap is required for DimensionalityReductionAlgorithm.TRIMAP. Install via pip install trimap") from e

                    reducer = trimap.TRIMAP(n_dims=config.dimensionality_reduction_n_components, random_state=42)
                    X_reduced = reducer.fit_transform(X)

                else:
                    raise ValueError(f"Unsupported dimensionality reduction algorithm: {config.dimensionality_reduction_algorithm}")

                # Zapisz dopasowany reduktor tylko jeśli wspiera transform (UMAP wspiera transform())
                try:
                    # Sprawdź czy reducer ma metodę transform
                    if hasattr(reducer, 'transform'):
                        self._last_reducer = reducer
                        self._last_reducer_config = config_key
                        print("Stored fitted reducer for reuse on test set.")
                    else:
                        # Nie nadpisuj istniejącego reduktora dla algorytmów bez transform()
                        print("Reducer does not support transform(); it will not be reused for test set.")
                        self._last_reducer = None
                        self._last_reducer_config = None
                except NameError:
                    # Jeśli reducer nie został utworzony (np. transform próby użycia wcześniej), po prostu nie zapisuj
                    self._last_reducer = None
                    self._last_reducer_config = None

            try:
                print(f"Reduced dimensions from {X.shape[1]} to {X_reduced.shape[1]}")
            except Exception:
                pass
            return X_reduced

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

        with open(file_path, 'r', encoding='utf-8') as file:
            first_line = file.readline().strip()
        has_header = bool(first_line) and not (first_line[0].isdigit() or first_line[0] in '+-')

        data = np.loadtxt(file_path, delimiter=',', skiprows=1 if has_header else 0)
        if data.ndim == 1:
            data = data.reshape(1, -1)

        labels = data[:, 0].astype(np.int64)
        features = data[:, 1:].astype(np.float64, copy=False)
        vectors = [
            VectorNumberData(label=int(labels[i]), vector=features[i].tolist())
            for i in range(features.shape[0])
        ]

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

        labels = np.asarray([v.label for v in vectors], dtype=np.int64).reshape(-1, 1)
        features = np.asarray([v.vector for v in vectors], dtype=np.float64)
        data = np.hstack((labels, features))

        header = ''
        if include_header:
            vector_size = features.shape[1]
            header = ','.join(['label'] + [f'feature_{i}' for i in range(vector_size)])

        np.savetxt(file_path, data, delimiter=',', header=header, comments='')

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

    def _meta_path(self) -> Path:
        """Returns path to the metadata file stored alongside the vectors CSV."""
        return Path(self.default_vectors_file).with_suffix('.meta.json')

    def _extract_config_key(self, config: Any) -> dict:
        """Extracts the fields from config that affect vector content."""
        flood_cfg = getattr(config, 'flood_config', None)
        if flood_cfg is not None:
            flood_str = flood_cfg if isinstance(flood_cfg, str) else flood_cfg.to_string()
        else:
            flood_str = None
        return {
            'dataset_name': getattr(config, 'dataset_name', None),
            'training_path': getattr(config, 'train_path', None),
            'test_path': getattr(config, 'test_path', None),
            'class_count': getattr(config, 'class_count', None),
            'pixel_normalization_rate': getattr(config, 'pixel_normalization_rate', None),
            'num_segments': getattr(config, 'num_segments', None),
            'flood_config': flood_str,
            'training_set_limit': getattr(config, 'training_set_limit', None),
            'dimensionality_reduction_algorithm': str(getattr(config, 'dimensionality_reduction_algorithm', None)),
            'dimensionality_reduction_n_components': getattr(config, 'dimensionality_reduction_n_components', None),
            'image_size': getattr(config, 'image_size', None),
        }

    def _save_config_metadata(self, config: Any) -> None:
        """Saves the config key as JSON metadata alongside the vectors CSV."""
        try:
            meta = self._extract_config_key(config)
            meta_path = self._meta_path()
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            with open(meta_path, 'w') as f:
                json.dump(meta, f)
        except Exception as e:
            print(f"Warning: Could not save vector metadata: {e}")

    def _load_config_metadata(self) -> Optional[dict]:
        """Loads the cached config key from JSON metadata file, or None if missing/corrupt."""
        meta_path = self._meta_path()
        if not meta_path.exists():
            return None
        try:
            with open(meta_path, 'r') as f:
                return json.load(f)
        except Exception:
            return None

    def get_test_vectors(
        self,
        raw_data: List[RawNumberData],
        config: Any
    ) -> List[VectorNumberData]:
        """Returns test vectors, using in-memory cache when config is unchanged."""
        current_key = self._extract_config_key(config)
        if self._cached_test_vectors is not None and self._last_test_config_key == current_key:
            print("Using cached test vectors.")
            return self._cached_test_vectors
        print("Preparing test vectors...")
        self._cached_test_vectors = self._prepare_vectors_batch(raw_data, config, is_training_data=False)
        self._last_test_config_key = current_key
        return self._cached_test_vectors

    def _can_load_from_file(self, new_config: Any) -> bool:
        """Sprawdza czy można wczytać wektory z pliku zamiast generować"""
        if not Path(self.default_vectors_file).exists():
            return False
        saved_key = self._load_config_metadata()
        if saved_key is None:
            return False
        current_key = self._extract_config_key(new_config)
        return saved_key == current_key
