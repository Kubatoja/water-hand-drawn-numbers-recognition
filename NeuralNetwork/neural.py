import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Funkcja do wczytywania danych
def load_data(train_file, test_file):
    # Wczytaj dane treningowe (pomiń nagłówki, jeśli istnieją)
    train_data = pd.read_csv(train_file, header=None)
    X_train = train_data.iloc[:, 1:].astype(np.float32).values
    y_train = train_data.iloc[:, 0].astype(np.uint8).values

    # Wczytaj dane testowe (pomiń nagłówki, jeśli istnieją)
    test_data = pd.read_csv(test_file, header=None)
    X_test = test_data.iloc[:, 1:].astype(np.float32).values
    y_test = test_data.iloc[:, 0].astype(np.uint8).values

    # Normalizacja danych (skalowanie do zakresu 0-1)
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # One-hot encoding dla etykiet
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return X_train, y_train, X_test, y_test

# Funkcja do trenowania modelu
def train_model(X_train, y_train):
    start_time = time.time()

    model = Sequential()
    model.add(Dense(128, input_shape=(784,), activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1)

    end_time = time.time()
    training_time = end_time - start_time
    print(f"Czas treningu: {training_time:.2f} sekund")

    return model, training_time

# Funkcja do testowania modelu
def test_model(model, X_test, y_test):
    start_time = time.time()

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    end_time = time.time()
    testing_time = end_time - start_time
    time_per_image = testing_time / len(X_test)

    print(f"Skuteczność modelu: {accuracy * 100:.2f}%")
    print(f"Czas testowania: {testing_time:.2f} sekund")
    print(f"Czas testowania jednej liczby: {time_per_image:.6f} sekund")

    return accuracy, testing_time, time_per_image

# Główna funkcja
def main():
    # Wczytanie danych
    X_train, y_train, X_test, y_test = load_data('usps_train.csv', 'usps_test.csv')

    # Trenowanie modelu
    model, training_time = train_model(X_train, y_train)

    # Testowanie modelu
    accuracy, testing_time, time_per_image = test_model(model, X_test, y_test)

if __name__ == "__main__":
    main()