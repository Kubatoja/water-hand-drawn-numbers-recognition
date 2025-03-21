from Tester.tester import generate_forest
import tkinter as tk


from draw.draw import DrawingApp

if __name__ == "__main__":
    # Generujemy las (ANN)
    forest = generate_forest()

    # Inicjalizacja aplikacji
    root = tk.Tk()
    app = DrawingApp(root, forest=forest)
    root.mainloop()