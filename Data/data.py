import numpy as np
import csv

# Loading data file = "test" or file = "train"
def load_data(file):
    test_data = 'Data/mnist_test.csv'  
    train_data = 'Data/mnist_train.csv'

    if file == "test":
        file_path = test_data
    elif file== "train": 
        file_path = train_data

    data = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the headlines
        for row in reader:
            data.append(row)
    data = np.array(data, dtype=float)  # Conversion for float
    labels = data[:, 0]  # The first column is a label of what number it is
    pixels = data[:, 1:] / 255.0  # Normalize pixels to the range [0, 1]
    return pixels, labels

def binarize_data(pixels):
    return np.where(pixels > 0.5, 1, 0)