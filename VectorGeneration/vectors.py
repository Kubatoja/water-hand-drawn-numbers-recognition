from Data.data import get_data
from BFS.bfs import *
from Data.data import load_data
def generate_training_vectors():
    dataset = "train"
    
    pixels, labels = load_data(dataset)
    trainingSetSize = 59999
    numSegments = 5
    pixelNormalizationRate = 0.34
    floodSides = "1111"

    print("Generating Vectors")
    generate_vectors_for_n(trainingSetSize, numSegments, pixels, labels, pixelNormalizationRate, floodSides=floodSides)
    print(f"Generated vectors for {trainingSetSize} numbers")


def create_vector_for_one_number(binarized_data, label, numSegments, floodSides="1111"):

    left_flooded, right_flooded, top_flooded, bottom_flooded, inverted_correction_array = flood_from_all_sides(binarized_data)

    flooded_vector = calculate_flooded_vector(binarized_data, left_flooded, right_flooded, top_flooded, bottom_flooded, inverted_correction_array, num_segments=numSegments, floodSides=floodSides)

    # flooded_vector.insert(0, label.flatten().tolist()[0]) # Add label to the beginning of the vector
    return flooded_vector

def generate_vectors_for_n(n, numSegments, pixel, labels, pixelNormalizationRate, floodSides="1111"):
    with open("Data/vectors.csv", 'w') as file:
        for i in range (0, n-1):
            binarized_data, label = get_data(pixel, labels, i, pixelNormalizationRate)
            file.write(str(create_vector_for_one_number(binarized_data, label, numSegments, floodSides=floodSides)).replace('[', '').replace(']','')+'\n')