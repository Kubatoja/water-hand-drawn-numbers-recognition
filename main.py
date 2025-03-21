from Tester.tester import generate_forest
import tkinter as tk
from VectorGeneration.vectors import generate_training_vectors

from draw.draw import DrawingApp

if __name__ == "__main__":


    # generate_training_vectors()
    forest = generate_forest()

    root = tk.Tk()
    app = DrawingApp(root, forest=forest)
    root.mainloop()