from Data.data import load_data, binarize_data, load_vectors
from VectorGeneration.vectors import *
from VectorSearch.knn import *
from VectorSearch.annoy import *
import csv
import os
import time
from datetime import datetime
import itertools


def generate_forest():
    treesCount = 2
    leavesCount = 328
    
    train_vectors, train_labels = load_vectors()

    print("Generating Forest")
    forest = build_forest(train_vectors, train_labels, treesCount, leavesCount, 0.95)
    print("Forest Generated")
    return forest

def test_vector(forest, vector):
    
    val = approximate_label(forest, vector, 10)
    return val

