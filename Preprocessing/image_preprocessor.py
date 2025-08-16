import numpy as np
from typing import Tuple


class ImagePreprocessor:
    """Klasa do preprocessing'u obrazów cyfr przed flood fill"""
    
    def __init__(self, target_size: int = 28):
        """
        Inicjalizuje preprocessor
        
        Args:
            target_size: Docelowy rozmiar obrazu (domyślnie 28x28)
        """
        self.target_size = target_size
    
    def center_digit(self, image: np.ndarray) -> np.ndarray:
        """
        Centruje cyfrę w obrazie poprzez wykrycie bounding box i przesunięcie
        
        Args:
            image: Obraz 2D (28x28) z binarną cyfrą (0 i 1)
            
        Returns:
            Obraz z wycentrowaną cyfrą
        """
        # Znajdź bounding box cyfry
        bbox = self._find_bounding_box(image)
        
        if bbox is None:
            # Jeśli nie znaleziono żadnych pikseli, zwróć oryginalny obraz
            return image.copy()
        
        # Wyciągnij cyfrę z bounding box
        min_row, max_row, min_col, max_col = bbox
        digit_height = max_row - min_row + 1
        digit_width = max_col - min_col + 1
        
        # Wyciągnij samą cyfrę
        digit = image[min_row:max_row+1, min_col:max_col+1]
        
        # Stwórz nowy pusty obraz
        centered_image = np.zeros((self.target_size, self.target_size))
        
        # Oblicz pozycję centrującą
        start_row = (self.target_size - digit_height) // 2
        start_col = (self.target_size - digit_width) // 2
        
        # Upewnij się, że cyfra zmieści się w obrazie
        end_row = min(start_row + digit_height, self.target_size)
        end_col = min(start_col + digit_width, self.target_size)
        
        # Umieść cyfrę w centrum
        centered_image[start_row:end_row, start_col:end_col] = digit[:end_row-start_row, :end_col-start_col]
        
        return centered_image
    
    def _find_bounding_box(self, image: np.ndarray) -> Tuple[int, int, int, int] | None:
        """
        Znajduje bounding box cyfry w obrazie
        
        Args:
            image: Obraz 2D z binarną cyfrą
            
        Returns:
            Tuple (min_row, max_row, min_col, max_col) lub None jeśli nie znaleziono cyfry
        """
        # Znajdź wszystkie niezerowe piksele
        nonzero_positions = np.where(image > 0)
        
        if len(nonzero_positions[0]) == 0:
            return None
        
        # Znajdź granice
        min_row = np.min(nonzero_positions[0])
        max_row = np.max(nonzero_positions[0])
        min_col = np.min(nonzero_positions[1])
        max_col = np.max(nonzero_positions[1])
        
        return min_row, max_row, min_col, max_col
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Główna metoda preprocessing'u - centruje cyfrę
        
        Args:
            image: Oryginalny obraz 28x28
            
        Returns:
            Przetworzony obraz z wycentrowaną cyfrą
        """
        return self.center_digit(image)
    
    def get_centering_stats(self, image: np.ndarray) -> dict:
        """
        Zwraca statystyki dotyczące centrowania dla debugowania
        
        Args:
            image: Obraz 2D
            
        Returns:
            Słownik ze statystykami
        """
        bbox = self._find_bounding_box(image)
        
        if bbox is None:
            return {
                "has_content": False,
                "bbox": None,
                "width": 0,
                "height": 0,
                "center_of_mass": None
            }
        
        min_row, max_row, min_col, max_col = bbox
        width = max_col - min_col + 1
        height = max_row - min_row + 1
        
        # Oblicz środek masy
        y_coords, x_coords = np.where(image > 0)
        if len(y_coords) > 0:
            center_y = np.mean(y_coords)
            center_x = np.mean(x_coords)
        else:
            center_y = center_x = None
        
        return {
            "has_content": True,
            "bbox": bbox,
            "width": width,
            "height": height,
            "center_of_mass": (center_y, center_x) if center_y is not None else None
        }
