from Visualization.visualize import display_image, visualize_flooded_number
from Tester.tester import *

import cProfile #remove later

def visualize(number_index):
    binarized_data, label = get_data(number_index)

    left_flooded, right_flooded, top_flooded, bottom_flooded, inverted_correction_array = flood_from_all_sides(binarized_data)

    visualize_flooded_number(binarized_data, left_flooded, right_flooded, top_flooded, bottom_flooded, inverted_correction_array)

def DEBUG_calculate_speed():
    for number_index in range(10000):
        print(number_index, "  ", create_vector_for_one_number(number_index=number_index))

if __name__ == '__main__':
    test()


    #visualize(number_index=number_index)
    # cProfile.run('DEBUG_calculate_speed()')

