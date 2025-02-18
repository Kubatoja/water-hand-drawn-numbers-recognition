import numpy as np
import csv

# wczytanie danych file = "test" lub file = "train"
def load_data(file):
    test_data = 'Dane\mnist_test.csv'  
    train_data = 'Dane\mnist_train.csv'

    if file == "test":
        file_path = test_data
    else:
        file_path = train_data

    data = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Pomijamy nagłówek
        for row in reader:
            data.append(row)
    data = np.array(data, dtype=float)  # Konwersja na float
    labels = data[:, 0]  # Pierwsza kolumna to etykieta jaka to liczba
    pixels = data[:, 1:] / 255.0  # Normalizacja pikseli do zakresu [0, 1]
    return pixels, labels

def binarize_data(pixels):
    return np.where(pixels > 0.5, 1, 0)