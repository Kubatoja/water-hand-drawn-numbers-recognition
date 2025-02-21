from collections import deque
import numpy as np

def bfs_flood_from_side(array, side):
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
        raise ValueError("Nieprawidłowa strona. Wybierz 'left', 'right', 'top' lub 'bottom'.")
    
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
    
    return filled

def flood_from_all_sides(array):
    left_flooded = bfs_flood_from_side(array, 'right')
    right_flooded = bfs_flood_from_side(array, 'left')
    
    # For top and bottom swap axes
    top_flooded = bfs_flood_from_side(array.T, 'right').T  
    bottom_flooded = bfs_flood_from_side(array.T, 'left').T 
    
    return left_flooded, right_flooded, top_flooded, bottom_flooded