import matplotlib.pyplot as plt
import numpy as np

def display_image(image_data, label):
    image = image_data.reshape(28, 28) 
    plt.imshow(image, cmap='gray')
    plt.title(f"Etykieta: {label}")
    plt.axis('off')  
    plt.show()


def visualize_flooded_number(original_array, left_flooded, right_flooded, top_flooded, bottom_flooded):
    rows, cols = original_array.shape
    images = [
        np.zeros((rows, cols, 3)),  
        np.zeros((rows, cols, 3)), 
        np.zeros((rows, cols, 3)), 
        np.zeros((rows, cols, 3)), 
        np.zeros((rows, cols, 3)) 
    ]
    
 
    images[0][original_array == 1] = [1, 1, 1]  # White color for number
    

    for i, flooded_array in enumerate([left_flooded, right_flooded, top_flooded, bottom_flooded], start=1):
        water_mask = (flooded_array == 0)  
        images[i][water_mask] = [0.27, 0.55, 0.85] # Blue color for water
        images[i][original_array == 1] = [1, 1, 1] # White color for number
    
    titles = [
        "Og number",
        "Water from the left",
        "Water from the right",
        "Water from the top",
        "Water from the bottom"
    ]
    
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')  
    plt.tight_layout()
    plt.show()