import numpy as np
import pandas as pd
import cv2
from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle

def center_on_canvas(image, canvas_size=28):
    """Umieszcza obraz 16x16 na środku obrazu 28x28 (czarne tło)."""
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    y_offset = (canvas_size - image.shape[0]) // 2
    x_offset = (canvas_size - image.shape[1]) // 2
    canvas[y_offset:y_offset+image.shape[0], x_offset:x_offset+image.shape[1]] = image
    return canvas

def rescale_to_255(image):
    # USPS images are w [-1, 1]; map to [0, 255]
    return ((image + 1) * 127.5).astype(np.uint8)

def main():
    print("Fetching USPS dataset from OpenML...")
    usps = fetch_openml("usps", version=1, as_frame=False)
    images, labels = usps['data'], usps['target']
    
    print("Normalizing and centering images...")
    images = images.reshape(-1, 16, 16)
    images_rescaled = np.array([rescale_to_255(img) for img in images])
    images_centered = np.array([center_on_canvas(img) for img in images_rescaled])
    
    # Flatten each image and convert labels to integers
    images_flat = images_centered.reshape(len(images_centered), -1)
    labels = labels.astype(int) - 1

    print("Shuffling and splitting dataset...")
    X, y = shuffle(images_flat, labels, random_state=42)
    
    # Create training and test splits
    X_train, y_train = X[:8000], y[:8000]
    X_test, y_test = X[8000:], y[8000:]

    print("Saving to CSV...")
    train_data = np.column_stack((y_train, X_train))
    test_data = np.column_stack((y_test, X_test))
    
    np.savetxt("usps_train.csv", train_data, fmt='%d', delimiter=",")
    np.savetxt("usps_test.csv", test_data, fmt='%d', delimiter=",")

    print("Done! Saved 'usps_train.csv' and 'usps_test.csv'.")

if __name__ == "__main__":
    main()
