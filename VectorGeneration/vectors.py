from Data.data import get_data
from BFS.bfs import *
def create_vector_for_one_number(binarized_data, label, numSegments):

    left_flooded, right_flooded, top_flooded, bottom_flooded, inverted_correction_array = flood_from_all_sides(binarized_data)

    flooded_vector = calculate_flooded_vector(binarized_data, left_flooded, right_flooded, top_flooded, bottom_flooded, inverted_correction_array, num_segments=numSegments)
    flooded_vector.insert(0, label.flatten().tolist()[0]) # Add label to the beginning of the vector

    # label and 9 features
    return flooded_vector

def generate_vectors_for_n(n, numSegments, pixel, labels, pixelNormalizationRate):
    with open("Data/vectors.csv", 'w') as file:
        for i in range (0, n-1):
            binarized_data, label = get_data(pixel, labels, i, pixelNormalizationRate)
            file.write(str(create_vector_for_one_number(binarized_data, label, numSegments)).replace('[', '').replace(']','')+'\n')