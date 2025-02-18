from Wizualizacje.wizualizacja import display_image
from Dane.dane import load_data, binarize_data

def wizualizuj():
    pixels, labels = load_data("test")

    data_number = 1
    image_data = pixels[data_number]  
    label = labels[data_number] 

    print(binarize_data(image_data).reshape(28, 28))
    display_image(image_data, label)

if __name__ == '__main__':
    wizualizuj()