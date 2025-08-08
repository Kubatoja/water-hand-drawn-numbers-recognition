import numpy as np
import time


def load_data(filename):
    data = np.genfromtxt(filename, delimiter=',')
    labels = data[:, 0].astype(int)
    pixels = data[:, 1:] / 255.0  # Normalizacja pikseli do zakresu [0, 1]
    return labels, pixels


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


def cosine_similarity(x1, x2):
    # Im większa wartość, tym bardziej podobne (używamy 1 - similarity dla spójności z odległością)
    dot_product = np.dot(x1, x2)
    norm_x1 = np.sqrt(np.sum(x1 ** 2))
    norm_x2 = np.sqrt(np.sum(x2 ** 2))
    return dot_product / (norm_x1 * norm_x2)


class KNN:
    def __init__(self, k=3, metric='euclidean'):
        self.k = k
        self.metric = metric

        # Słownik mapujący nazwę metryki na odpowiednią funkcję
        self.metric_functions = {
            'euclidean': euclidean_distance,
            'cosine': lambda x, y: 1 - cosine_similarity(x, y)  # Konwersja podobieństwa na "odległość"
        }

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, x):
        metric_func = self.metric_functions[self.metric]
        # Oblicz odległości od wszystkich punktów treningowych
        distances = [metric_func(x, x_train) for x_train in self.X_train]

        # Znajdź k najbliższych sąsiadów
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # Głosowanie - najczęstsza etykieta
        most_common = np.bincount(k_nearest_labels).argmax()

        return most_common


def evaluate_accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy * 100
