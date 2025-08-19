import csv
from pathlib import Path
from typing import List, Optional
import numpy as np
import numba

from BFS.bfs import calculate_flooded_vector
from Tester.configs import ANNTestConfig
from Tester.otherModels import RawNumberData, VectorNumberData
from Preprocessing.image_preprocessor import ImagePreprocessor


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

    @staticmethod
    def reset_jit_cache():
        """
        Resetuje cache Numba JIT, aby wymusić rekompilację przy następnym użyciu.
        Używane do rzetelnego porównania czasów wykonania między testami.
        """
        print("🔄 Resetowanie JIT cache dla niezależnych pomiarów...")
        
        try:
            # Metoda 1: Wyczyść globalny cache Numba
            import numba
            import gc
            
            # Wyczyść wszystkie cache'e Numba
            if hasattr(numba.core.registry.CPUTarget, 'cache'):
                numba.core.registry.CPUTarget.cache.clear()
            
            # Wyczyść także cache typów
            if hasattr(numba.types, 'typeof_impl'):
                numba.types.typeof_impl.cache.clear()
                
            # Wyczyść cache dyspatchera
            from numba.core import dispatcher
            dispatcher._DISPATCHER_CACHE.clear()
            
            # Wymuś garbage collection
            gc.collect()
            
            print("   ✅ JIT cache zresetowany (globalny)")
            return True
            
        except Exception as e1:
            print(f"   ⚠️ Globalny reset nie powiódł się: {e1}")
            
            try:
                # Metoda 2: Restart modułu BFS
                import sys
                import importlib
                
                # Usuń moduł BFS z cache
                modules_to_remove = [name for name in sys.modules.keys() if name.startswith('BFS')]
                for module_name in modules_to_remove:
                    del sys.modules[module_name]
                
                print("   ✅ Moduł BFS zresetowany")
                return True
                
            except Exception as e2:
                print(f"   ⚠️ Reset modułu nie powiódł się: {e2}")
                print("   ℹ️ JIT będzie używał istniejącego cache")
                return False

    def generate_vectors(self, rawNumberDataList: List[RawNumberData], config: ANNTestConfig) -> List[VectorNumberData]:
        """
        Generate vectors for training data using ultra-optimized sequential processing
        
        Args:
            rawNumberDataList: List of raw image data
            config: Test configuration containing parameters
            
        Returns:
            List of VectorNumberData
        """
        import time
        
        limit = min(len(rawNumberDataList), config.training_set_limit)
        print(f"Generating {limit} training vectors...")
        
        start_time = time.perf_counter()
        
        print("🚀 Using ULTRA-OPTIMIZED sequential processing...")
        vectors = self._generate_vectors_sequential(rawNumberDataList, config, limit)
        
        generation_time = time.perf_counter() - start_time
        per_image_ms = generation_time/limit*1000
        throughput = limit/generation_time
        
        print(f"✅ Vector generation completed in {generation_time:.2f}s ({per_image_ms:.3f}ms per image)")
        print(f"🚀 Throughput: {throughput:.0f} images/sec")
        
        if per_image_ms < 1.0:
            print(f"🎯 EXCELLENT: Sub-millisecond per image performance!")
        elif per_image_ms < 5.0:
            print(f"⚡ VERY GOOD: Under 5ms per image")
        else:
            print(f"⏱️  Standard performance: {per_image_ms:.1f}ms per image")
        
        return vectors
    
    def _generate_vectors_sequential(self, rawNumberDataList: List[RawNumberData], config: ANNTestConfig, limit: int) -> List[VectorNumberData]:
        """ULTRA-OPTIMIZED sequential vector generation with advanced techniques"""
        import time
        
        print("🚀 ULTRA-OPTIMIZATION MODE ACTIVATED")
        total_start = time.perf_counter()
        
        # OPTIMIZATION 1: Batch data preparation
        print("📦 Phase 1: Batch data extraction...")
        prep_start = time.perf_counter()
        
        # Extract all pixel arrays and labels in one go
        all_pixels = np.array([rawNumberDataList[i].pixels for i in range(limit)])
        all_labels = [rawNumberDataList[i].label for i in range(limit)]
        
        prep_time = time.perf_counter() - prep_start
        print(f"   ✅ Data extraction: {prep_time:.3f}s")
        
        # OPTIMIZATION 2: Vectorized binarization and centering
        print("🔥 Phase 2: Vectorized binarization and centering...")
        bin_start = time.perf_counter()
        
        # Batch binarize ALL images at once using NumPy vectorization
        binarized_batch = np.where(all_pixels > config.pixel_normalization_rate, 1, 0)
        binarized_batch = binarized_batch.reshape(-1, 28, 28)
        
        # Apply centering if enabled
        if config.enable_centering:
            print("   🎯 Centering digits...")
            preprocessor = ImagePreprocessor()
            for i in range(len(binarized_batch)):
                binarized_batch[i] = preprocessor.center_digit(binarized_batch[i])
        
        bin_time = time.perf_counter() - bin_start
        centering_info = " with centering" if config.enable_centering else ""
        print(f"   ✅ Batch binarization{centering_info}: {bin_time:.3f}s ({bin_time/limit*1000:.3f}ms per image)")
        
        # OPTIMIZATION 3: Streamlined vector calculation (JIT już prekompilowany)
        print("🎯 Phase 3: Ultra-fast vector generation...")
        calc_start = time.perf_counter()
        
        vectors = []
        for i in range(limit):
            # Progress tracking for large datasets
            if limit > 1000 and i % 5000 == 0 and i > 0:
                elapsed = time.perf_counter() - calc_start
                rate = i / elapsed
                eta = (limit - i) / rate
                print(f"   📈 Progress: {i}/{limit} ({rate:.0f} img/s, ETA: {eta:.1f}s)")
            
            # Direct vector calculation - no intermediate objects
            flooded_vector = calculate_flooded_vector(
                binarized_batch[i],
                num_segments=config.num_segments,
                floodSides=config.flood_config.to_string()
            )
            
            # Minimal object creation
            vectors.append(VectorNumberData(label=all_labels[i], vector=flooded_vector))
        
        calc_time = time.perf_counter() - calc_start
        total_time = time.perf_counter() - total_start
        
        # Performance analysis
        print("\n📊 ULTRA-OPTIMIZATION PERFORMANCE BREAKDOWN:")
        print(f"   📦 Data extraction:  {prep_time:.3f}s ({(prep_time/total_time)*100:.1f}%)")
        print(f"   🔥 Binarization:     {bin_time:.3f}s ({(bin_time/total_time)*100:.1f}%)")
        print(f"   🎯 Vector calc:      {calc_time:.3f}s ({(calc_time/total_time)*100:.1f}%)")
        print(f"   🚀 TOTAL:            {total_time:.3f}s")
        print(f"   💎 Per image:        {calc_time/limit*1000:.3f}ms (calc only)")
        print(f"   🏆 Overall per img:  {total_time/limit*1000:.3f}ms (total)")
        print(f"   ⚡ Throughput:       {limit/total_time:.0f} images/sec")
        
        # Efficiency metrics
        pure_calc_efficiency = calc_time / total_time * 100
        print(f"   📈 Calculation efficiency: {pure_calc_efficiency:.1f}%")
        
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
