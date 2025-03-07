from collections import deque
import numpy as np

def bfs_flood_from_side(array, side, allow_backtrack=False):
    rows, cols = array.shape
    filled = np.copy(array)
    
    if side == 'left':
        start_col = 0
        direction = (0, 1)  # Water flows from left to right
    elif side == 'right':
        start_col = cols - 1
        direction = (0, -1)  # Water flows from right to left
    elif side == 'top':
        start_row = 0
        direction = (1, 0)  # Water flows from top to bottom
    elif side == 'bottom':
        start_row = rows - 1
        direction = (-1, 0)  # Water flows from the bottom to the top
    else:
        raise ValueError("Wrong side. Choose 'left', 'right', 'top' or 'bottom'.")
    
    queue = deque()
    visited = set()
    
    if side in ['left', 'right']:
        for i in range(rows):
            if array[i, start_col] == 0:
                queue.append((i, start_col))
                visited.add((i, start_col))
    elif side in ['top', 'bottom']:
        for j in range(cols):
            if array[start_row, j] == 0:
                queue.append((start_row, j))
                visited.add((start_row, j))
    
    while queue:
        row, col = queue.popleft()
        filled[row, col] = 1  
        
        for dr, dc in [(-1, 0), (1, 0), direction]:
            new_row, new_col = row + dr, col + dc
            if (0 <= new_row < rows and 0 <= new_col < cols and
                (new_row, new_col) not in visited and
                filled[new_row, new_col] == 0):
               
                if side == 'left' and new_col >= start_col:
                    queue.append((new_row, new_col))
                    visited.add((new_row, new_col))
                elif side == 'right' and new_col <= start_col:
                    queue.append((new_row, new_col))
                    visited.add((new_row, new_col))
                elif side == 'top' and new_row >= start_row:
                    queue.append((new_row, new_col))
                    visited.add((new_row, new_col))
                elif side == 'bottom' and new_row <= start_row:
                    queue.append((new_row, new_col))
                    visited.add((new_row, new_col))
        
        # Allow backtracking if the flag is set
        if allow_backtrack:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_row, new_col = row + dr, col + dc
                if (0 <= new_row < rows and 0 <= new_col < cols and
                    (new_row, new_col) not in visited and
                    filled[new_row, new_col] == 0):
                    queue.append((new_row, new_col))
                    visited.add((new_row, new_col))
    
    return filled

def flood_from_all_sides(array):
    left_flooded = bfs_flood_from_side(array, 'right')
    right_flooded = bfs_flood_from_side(array, 'left')
    
    # For top and bottom swap axes
    top_flooded = bfs_flood_from_side(array.T, 'right').T  
    bottom_flooded = bfs_flood_from_side(array.T, 'left').T 

    correction_array = bfs_flood_from_side(array, 'left', allow_backtrack=True)

    inverted_correction_array = 1 - correction_array
    
    return left_flooded, right_flooded, top_flooded, bottom_flooded, inverted_correction_array

def calculate_flooded_vector(original_array, left_flooded, right_flooded, top_flooded, bottom_flooded, inverted_correction_array, num_segments=2, floodSides="1111"):
    rows, cols = original_array.shape
    
    def split_array(array, num_segments):
        segment_height = rows // num_segments
        segments = []
        for i in range(num_segments):
            start_row = i * segment_height
            end_row = (i + 1) * segment_height if i < num_segments - 1 else rows
            segments.append(array[start_row:end_row, :])
        return segments
    
    correction_segments = split_array(inverted_correction_array, num_segments)
    
    correction_counts = [np.sum(segment == 1) for segment in correction_segments]
    
    flooded_arrays = []

    if floodSides[0] == "1":
        flooded_arrays.append(left_flooded)
    if floodSides[1] == "1":
        flooded_arrays.append(right_flooded)
    if floodSides[2] == "1":
        flooded_arrays.append(top_flooded)
    if floodSides[3] == "1":
        flooded_arrays.append(bottom_flooded)
    
    flooded_segments_list = [split_array(array, num_segments) for array in flooded_arrays]

    result_vector = []
    
    for flooded_segments in flooded_segments_list:
       
        for i in range(num_segments):
            zero_count = np.sum(flooded_segments[i] == 0)
            corrected_zero_count = zero_count - correction_counts[i]
            segment_size = flooded_segments[i].size
            if segment_size == 0:
                result_vector.append(0.0) 
            else:
                result_vector.append(max(0, corrected_zero_count) / segment_size)
    
    inverted_correction_array_segments = num_segments

    parts = np.array_split(inverted_correction_array, inverted_correction_array_segments)

    for part in parts:
        part_sum = part.sum()
        part_size = part.size
        normalized_part_sum = part_sum / part_size
        result_vector.append(normalized_part_sum)

    # result_vector.append(inverted_correction_array.sum() / 784.0)

    perimeter_array = calculate_perimeter(original_array, inverted_correction_array, num_segments)
    for perimeter in perimeter_array:
        result_vector.append(perimeter)

    return np.array(result_vector).flatten().tolist()


def calculate_perimeter(original_array, corrected_array_inverted, num_segments):
    combined_array = np.zeros_like(original_array)
    combined_array[(original_array == 1) | (corrected_array_inverted == 1)] = 1
    
    perimeter = 0
    visited = np.zeros_like(combined_array)  
    
    

    rows, cols = combined_array.shape
    for i in range(rows):
        for j in range(cols):
            if combined_array[i, j] == 1: 
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj

                    if ni < 0 or ni >= rows or nj < 0 or nj >= cols:
                        continue 

                    if combined_array[ni, nj] == 0 and not visited[ni, nj]:
                        perimeter += 1
                        visited[ni, nj] = 1 

    perimeter_array = []
    parts = np.array_split(visited, num_segments)
    for part in parts:
        part_sum = part.sum()
        part_size = part.size
        normalized_part_sum = part_sum / part_size
        perimeter_array.append(normalized_part_sum)

    return perimeter_array