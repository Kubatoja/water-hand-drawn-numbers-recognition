import numpy as np
from Data.data import load_vectors

def euclidean_distance(vec1, vec2):
    squared_diffs = (vec1 - vec2) ** 2  # Element-wise subtraction and squaring
    sum_squared_diffs = np.sum(squared_diffs)  # Sum of the squared differences
    return np.sqrt(sum_squared_diffs)

def manhattan_distance(vec1, vec2):
    return np.sum(np.abs(vec1 - vec2))

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    magnitude1 = np.linalg.norm(vec1)
    magnitude2 = np.linalg.norm(vec2)
    if magnitude1 == 0 or magnitude2 == 0:
        return 0
    return 1 - (dot_product / (magnitude1 * magnitude2))

def get_most_common(arr):
    values, counts = np.unique(arr, return_counts=True)
    max_count = np.max(counts)
    most_common_value = values[counts == max_count][0]

    return most_common_value
def knn(new_vector, train_labels, train_vectors, k, distance_method):
    distances = []

    for train_vector in train_vectors:
        if distance_method == 1:
            dist = euclidean_distance(train_vector, new_vector)
        elif distance_method == 2:
            dist = manhattan_distance(train_vector, new_vector)
        elif distance_method == 3:
            dist = cosine_similarity(train_vector, new_vector)
        distances.append(dist)

    nearest_indices = np.argsort(distances)[:k]
    nearest_values = train_labels[nearest_indices]
    return get_most_common(nearest_values)
