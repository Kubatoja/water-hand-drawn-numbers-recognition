import tkinter as tk
from tkinter import font
import numpy as np

from VectorGeneration.vectors import generate_training_vectors, create_vector_for_one_number
from Tester.tester import test_vector

class DrawingApp:
    def __init__(self, root, forest, size=28, pixel_size=20):
        self.root = root
        self.size = size
        self.pixel_size = pixel_size
        self.canvas_size = size * pixel_size
        self.drawing = np.zeros((size, size), dtype=np.int8)  # Tablica binarna (0 i 1)
        self.forest = forest  # Przekazany wytrenowany las

        # Etykieta wyświetlająca przewidywaną liczbę
        self.approximated_number_label = tk.Label(root, text="Approximated number: --", font=font.Font(size=12))
        self.approximated_number_label.pack()

        # Canvas do rysowania
        self.canvas = tk.Canvas(root, width=self.canvas_size, height=self.canvas_size, bg="white")
        self.canvas.pack()

        # Przycisk "Get Array"
        self.button = tk.Button(root, text="Get Array", command=self.get_array)
        self.button.pack()

        # Przycisk "Reset"
        self.reset_button = tk.Button(root, text="Reset", command=self.reset_canvas)
        self.reset_button.pack()

        # Obsługa rysowania
        self.canvas.bind("<B1-Motion>", self.paint)

    def paint(self, event):
        """
        Obsługa rysowania na planszy.
        """
        x = event.x // self.pixel_size
        y = event.y // self.pixel_size
        if 0 <= x < self.size and 0 <= y < self.size:
            self.drawing[y, x] = 1  # Ustawiamy wartość na 1 (czarny)
            self.canvas.create_rectangle(
                x * self.pixel_size, y * self.pixel_size,
                (x + 1) * self.pixel_size, (y + 1) * self.pixel_size,
                fill="black", outline="black"
            )

    def get_array(self):
        """
        Przetwarza narysowaną liczbę i przewiduje jej etykietę za pomocą wytrenowanego lasu.
        """
        # Spłaszczamy tablicę do jednowymiarowej
        flat_array = self.drawing.flatten()

        # Przekształcamy z powrotem do kształtu 28x28
        reshaped_array = flat_array.reshape((self.size, self.size))

        # Tworzymy wektor dla narysowanej liczby
        vector = create_vector_for_one_number(reshaped_array,0.0, 5, floodSides="1111")
        print(vector)   
        # Przewidujemy etykietę za pomocą wytrenowanego lasu
        approximated_number = test_vector(forest=self.forest, vector=vector)

        # Aktualizujemy etykietę z przewidywaną liczbą
        self.approximated_number_label.config(text=f"Approximated number: {approximated_number}", fg="green")

    def reset_canvas(self):
        """
        Resetuje płótno i przywraca tekst u góry na "--".
        """
        self.drawing = np.zeros((self.size, self.size), dtype=np.int8)  # Resetujemy tablicę
        self.canvas.delete("all")  # Czyścimy płótno
        self.approximated_number_label.config(text="Approximated number: --", fg="black")  # Resetujemy etykietę


