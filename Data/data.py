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

def get_data(pixels, labels, data_index):

    image_data = pixels[data_index]
    label = labels[data_index]

    binarized_data = binarize_data(image_data).reshape(28, 28)

    return binarized_data, label
def load_vectors():
    path = 'Data/vectors.csv'

    labels = []
    data = []
    with open(path, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            data.append(row[1:])
            labels.append(row[0])
    data = np.array(data, dtype=float)  # Conversion for float
    labels = np.array(labels, dtype=float)
    return labels, data

def binarize_data(pixels):
    return np.where(pixels > 0.5, 1, 0)