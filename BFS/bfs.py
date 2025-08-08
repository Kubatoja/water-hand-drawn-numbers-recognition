import numpy as np
from collections import deque
from numba import njit  # Add this import if you can use numba

def bfs_flood_from_side(array, side, allow_backtrack=False):
    rows, cols = array.shape
    filled = np.copy(array)
    
    # Pre-compute directions
    if side == 'left':
        start_col = 0
        main_direction = (0, 1)
    elif side == 'right':
        start_col = cols - 1
        main_direction = (0, -1)
    elif side == 'top':
        start_row = 0
        main_direction = (1, 0)
    elif side == 'bottom':
        start_row = rows - 1
        main_direction = (-1, 0)
    else:
        raise ValueError("Invalid side. Choose 'left', 'right', 'top' or 'bottom'.")
    
    queue = deque()
    visited = np.zeros_like(array, dtype=bool)
    
    # Initialize queue based on side
    if side in ['left', 'right']:
        col = start_col
        for row in range(rows):
            if array[row, col] == 0:
                queue.append((row, col))
                visited[row, col] = True
    else:  # top or bottom
        row = start_row
        for col in range(cols):
            if array[row, col] == 0:
                queue.append((row, col))
                visited[row, col] = True
    
    # Pre-define all possible directions
    if allow_backtrack:
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        directions = [(-1, 0), (1, 0), main_direction]
    
    while queue:
        row, col = queue.popleft()
        filled[row, col] = 1
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            
            if (0 <= new_row < rows and 0 <= new_col < cols and 
                not visited[new_row, new_col] and 
                filled[new_row, new_col] == 0):
                
                # Check direction constraints if not backtracking
                if not allow_backtrack:
                    if side == 'left' and new_col < start_col:
                        continue
                    elif side == 'right' and new_col > start_col:
                        continue
                    elif side == 'top' and new_row < start_row:
                        continue
                    elif side == 'bottom' and new_row > start_row:
                        continue
                
                queue.append((new_row, new_col))
                visited[new_row, new_col] = True
    
    return filled

def flood_from_all_sides(array):
    # Process all sides in parallel (if using multiprocessing)
    left_flooded = bfs_flood_from_side(array, 'right')
    right_flooded = bfs_flood_from_side(array, 'left')
    top_flooded = bfs_flood_from_side(array.T, 'right').T
    bottom_flooded = bfs_flood_from_side(array.T, 'left').T
    correction_array = bfs_flood_from_side(array, 'left', allow_backtrack=True)
    
    return left_flooded, right_flooded, top_flooded, bottom_flooded, 1 - correction_array

def calculate_flooded_vector(original_array, num_segments=2, floodSides="1111"):
    # Get all flooded arrays at once
    left_flooded, right_flooded, top_flooded, bottom_flooded, inverted_correction_array = flood_from_all_sides(original_array)
    rows, cols = original_array.shape
    result_vector = []
    
    # Pre-compute correction segments
    correction_segments = np.array_split(inverted_correction_array, num_segments, axis=0)
    correction_counts = [np.sum(segment == 1) for segment in correction_segments]
    
    # Process flooded arrays
    flooded_arrays = []
    if floodSides[0] == "1":
        flooded_arrays.append(left_flooded)
    if floodSides[1] == "1":
        flooded_arrays.append(right_flooded)
    if floodSides[2] == "1":
        flooded_arrays.append(top_flooded)
    if floodSides[3] == "1":
        flooded_arrays.append(bottom_flooded)
    
    # Process all flooded arrays at once
    for array in flooded_arrays:
        segments = np.array_split(array, num_segments, axis=0)
        for i, segment in enumerate(segments):
            zero_count = np.sum(segment == 0)
            corrected_zero_count = max(0, zero_count - correction_counts[i])
            result_vector.append(corrected_zero_count / segment.size)
    
    # Add correction array features
    for part in correction_segments:
        result_vector.append(np.sum(part) / part.size)
    
    # Add original array segment features
    original_segments = np.array_split(original_array, num_segments, axis=0)
    for segment in original_segments:
        result_vector.append(np.sum(segment) / segment.size)
    
    # Add perimeter feature
    perimeter = calculate_perimeter(original_array, inverted_correction_array)
    result_vector.append(perimeter)
    
    return np.array(result_vector).flatten().tolist()

@njit  # Remove if you can't use numba
def calculate_perimeter(original_array, corrected_array_inverted):
    combined = (original_array == 1) | (corrected_array_inverted == 1)
    rows, cols = combined.shape
    perimeter = 0
    
    for i in range(rows):
        for j in range(cols):
            if combined[i, j]:
                # Check 4 neighbors
                if i > 0 and not combined[i-1, j]:
                    perimeter += 1
                elif i < rows-1 and not combined[i+1, j]:
                    perimeter += 1
                elif j > 0 and not combined[i, j-1]:
                    perimeter += 1
                elif j < cols-1 and not combined[i, j+1]:
                    perimeter += 1
    
    return perimeter / (rows * cols)  # Normalize by total pixels

def calculate_segments_value(original_array, num_segments):
    segments = np.array_split(original_array, num_segments, axis=0)
    return [np.sum(segment) / segment.size for segment in segments]