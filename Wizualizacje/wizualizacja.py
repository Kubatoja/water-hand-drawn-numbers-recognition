import matplotlib.pyplot as plt

def display_image(image_data, label):
    image = image_data.reshape(28, 28) 
    plt.imshow(image, cmap='gray')
    plt.title(f"Etykieta: {label}")
    plt.axis('off')  
    plt.show()
