from Visualization.visualize import display_image, visualize_flooded_number
from Data.data import load_data, binarize_data
from BFS.bfs import bfs_flood_from_side, flood_from_all_sides, calculate_flooded_vector

import cProfile #remove later

pixels, labels = load_data("test")

def get_data(data_number=22):

    image_data = pixels[data_number]  
    label = labels[data_number] 

    binarized_data = binarize_data(image_data).reshape(28, 28)

    return binarized_data, label

def visualize(number_index):
    binarized_data, label = get_data(number_index)

    left_flooded, right_flooded, top_flooded, bottom_flooded, inverted_correction_array = flood_from_all_sides(binarized_data)

    visualize_flooded_number(binarized_data, left_flooded, right_flooded, top_flooded, bottom_flooded, inverted_correction_array)

def create_vector_for_one_number(number_index):
    binarized_data, label = get_data(number_index)

    left_flooded, right_flooded, top_flooded, bottom_flooded, inverted_correction_array = flood_from_all_sides(binarized_data)

    flooded_vector = calculate_flooded_vector(binarized_data, left_flooded, right_flooded, top_flooded, bottom_flooded, inverted_correction_array, num_segments=2)
    flooded_vector.insert(0, label.flatten().tolist()[0]) # Add label to the beginning of the vector

    # label and 9 features
    return flooded_vector

def DEBUG_calculate_speed():
    for number_index in range(10000):
        print(number_index, "  ", create_vector_for_one_number(number_index=number_index))

if __name__ == '__main__':

    number_index = 22
    print(create_vector_for_one_number(number_index=number_index))
    visualize(number_index=number_index)


    # cProfile.run('DEBUG_calculate_speed()')

