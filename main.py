from Visualization.visualize import display_image, visualize_flooded_number
from Tester.tester import *

## Delete later
from Data.data import load_data, binarize_data
from BFS.bfs import bfs_flood_from_side, flood_from_all_sides, calculate_flooded_vector
import cProfile  


def visualize(number_index):
    pixels, labels = load_data("test")
    binarized_data, label = get_data(pixels, labels, number_index, pixelNormalizationRate=0.34)

    left_flooded, right_flooded, top_flooded, bottom_flooded, inverted_correction_array = flood_from_all_sides(binarized_data)

    visualize_flooded_number(binarized_data, left_flooded, right_flooded, top_flooded, bottom_flooded, inverted_correction_array)

def create_vector_for_one_number(number_index):
    pixels, labels = load_data("test")
    binarized_data, label = get_data(pixels, labels, number_index, pixelNormalizationRate=0.34)

    left_flooded, right_flooded, top_flooded, bottom_flooded, inverted_correction_array = flood_from_all_sides(binarized_data)

    flooded_vector = calculate_flooded_vector(binarized_data, left_flooded, right_flooded, top_flooded, bottom_flooded, inverted_correction_array, num_segments=2, floodSides="1001")
    flooded_vector.insert(0, label.flatten().tolist()[0]) # Add label to the beginning of the vector

    # label and 9 features
    return flooded_vector

def testVector(data_index):
    print(create_vector_for_one_number(data_index))
    visualize(data_index)


if __name__ == "__main__":
    date_string = str(datetime.now()).replace(' ', "").replace(':', '_')
    test(date_string, mode="ann")
    # testVector(65)
    # visualize(number_index=number_index)
    # cProfile.run('DEBUG_calculate_speed()')
