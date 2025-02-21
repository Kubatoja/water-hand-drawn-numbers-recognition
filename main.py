from Visualization.visualize import display_image, visualize_flooded_number
from Data.data import load_data, binarize_data
from BFS.bfs import bfs_flood_from_side, flood_from_all_sides

def visualize():
    pixels, labels = load_data("test")

    data_number = 123
    image_data = pixels[data_number]  
    label = labels[data_number] 

    binarized_data = binarize_data(image_data).reshape(28, 28)
    print(binarized_data)

    left_flooded, right_flooded, top_flooded, bottom_flooded = flood_from_all_sides(binarized_data)

    visualize_flooded_number(binarized_data, left_flooded, right_flooded, top_flooded, bottom_flooded)

if __name__ == '__main__':
    visualize()