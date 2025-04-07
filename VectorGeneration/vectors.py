from Data.data import get_data
from BFS.bfs import *
def create_vector_for_one_number(binarized_data, label, numSegments, floodSides="1111"):

    flooded_vector = calculate_flooded_vector(binarized_data, num_segments=numSegments, floodSides=floodSides)
    flooded_vector.insert(0, label.flatten().tolist()[0]) # Add label to the beginning of the vector

    # label and 9 features
    return flooded_vector

def generate_vectors_for_n(n, numSegments, pixel, labels, pixelNormalizationRate, floodSides="1111"):
    with open("Data/vectors.csv", 'w') as file:
        for i in range (0, n-1):
            binarized_data, label = get_data(pixel, labels, i, pixelNormalizationRate)
            file.write(str(create_vector_for_one_number(binarized_data, label, numSegments, floodSides=floodSides)).replace('[', '').replace(']','')+'\n')