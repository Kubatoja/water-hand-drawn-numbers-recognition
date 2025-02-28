import matplotlib.pyplot as plt
import numpy as np

def display_image(image_data, label):
    image = image_data.reshape(28, 28) 
    plt.imshow(image, cmap='gray')
    plt.title(f"Etykieta: {label}")
    plt.axis('off')  
    plt.show()


def visualize_flooded_number(original_array, left_flooded, right_flooded, top_flooded, bottom_flooded, inverted_correction_array):
    rows, cols = original_array.shape
    

    images = [
        np.zeros((rows, cols, 3)),  # Original number
        np.zeros((rows, cols, 3)),  # Water from the left
        np.zeros((rows, cols, 3)),  # Water from the right
        np.zeros((rows, cols, 3)),  # Water from the top
        np.zeros((rows, cols, 3)),  # Water from the bottom
        np.zeros((rows, cols, 3))   # Submarged number
    ]
    

    images[0][original_array == 1] = [1, 1, 1]  # White color for og number
    
    for i, flooded_array in enumerate([left_flooded, right_flooded, top_flooded, bottom_flooded], start=1):
        water_mask = (flooded_array == 0)  # Water areas
        images[i][water_mask] = [0.27, 0.55, 0.85]  # Blue color for water
        images[i][original_array == 1] = [1, 1, 1]  # White color for number
        images[i][inverted_correction_array == 1] = [0, 0, 0]  # Black color for correction areas
    

    images[5][:] = [0.27, 0.55, 0.85]  # Fill entire image with blue
    images[5][original_array == 1] = [1, 1, 1]  # White color for original number
    images[5][inverted_correction_array == 1] = [0, 0, 0]  # Black color for correction areas
    
    # Titles for each subplot
    titles = [
        "Original number",
        "Water from the left",
        "Water from the right",
        "Water from the top",
        "Water from the bottom",
        "Submerged number"
    ]
    

    fig, axes = plt.subplots(1, 6, figsize=(18, 3))
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')  # Hide axes
    plt.tight_layout()
    plt.show()