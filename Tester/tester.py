from Data.data import load_data, binarize_data, load_vectors
from VectorGeneration.vectors import *
from VectorSearch.knn import *
from VectorSearch.annoy import Ann
import csv
import os
import time
from datetime import datetime
import itertools

# testcases
# 1: Number of trees
# 2: Number of leaves
# 3: Training set size (1-9999)
# 4: num segments
# 5: pixelNormalizationRate
# 6: floodSides(Left, Right, Top, Bottom) STRING!!!

ANNtestCases = [
    [2, 328, 60000, 11, 0.34, "1111"],
    [2, 328, 60000, 12, 0.34, "1111"],
    [2, 328, 60000, 13, 0.34, "1111"],
    [2, 328, 60000, 14, 0.34, "1111"],
    [2, 328, 60000, 15, 0.34, "1111"],
    [2, 328, 60000, 16, 0.34, "1111"],
    [2, 328, 60000, 17, 0.34, "1111"],
    [2, 328, 60000, 18, 0.34, "1111"],
    [2, 328, 60000, 19, 0.34, "1111"],
    [2, 328, 60000, 20, 0.34, "1111"],
    ]


def generate_training_vectors(pixels, labels, trainingSetSize, numSegments, pixelNormalizationRate, floodSides="1111"):
    print("Generating Vectors")
    generate_vectors_for_n(trainingSetSize, numSegments, pixels, labels, pixelNormalizationRate, floodSides=floodSides)
    print(f"Generated vectors for {trainingSetSize} numbers")

    print("Loading Vectors")
    train_vectors, train_labels = load_vectors()
    print(f"Loaded {len(train_labels)} Vectors")
    return train_vectors, train_labels



def generate_csv_from_test_summary(test_summary, date, filename_prefix="test_results"):
    params_filename = f"{filename_prefix}_params_{date}.csv"
    matrices_filename = f"{filename_prefix}_matrices_{date}.csv"

    # Extract k_summaries and determine num_classes from the current test_summary's matrix
    k_summaries = test_summary[9]
    first_matrix = k_summaries[0][5]  # Assuming matrix is at index 5 in k_summary
    num_classes = len(first_matrix[0])

    # Check matrices file for existing num_classes if it exists
    matrices_file_exists = os.path.exists(matrices_filename)
    if matrices_file_exists:
        with open(matrices_filename, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            existing_num_classes = len(header) - 2  # Subtract TestID and Actual/Predicted columns
        if existing_num_classes != num_classes:
            raise ValueError(f"Matrices file expects {existing_num_classes} classes, current test has {num_classes} classes.")
    else:
        # Create matrices file with header
        with open(matrices_filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            header = ["TestID", "Actual / Predicted"] + [str(i) for i in range(num_classes)]
            writer.writerow(header)

    # Determine the starting test_id by reading existing params file
    params_file_exists = os.path.exists(params_filename)
    max_test_id = 0
    if params_file_exists:
        with open(params_filename, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                current_id = int(row[0])
                max_test_id = max(max_test_id, current_id)
    test_id = max_test_id + 1

    # Open files in append mode
    with open(params_filename, 'a', newline='', encoding='utf-8') as params_file, \
         open(matrices_filename, 'a', newline='', encoding='utf-8') as matrices_file:

        params_writer = csv.writer(params_file)
        matrices_writer = csv.writer(matrices_file)

        # Write params header if file is new
        if not params_file_exists:
            params_writer.writerow([
                "TestID",
                "Number of trees",
                "Number of members in leaf",
                "Training Set Size",
                "Testing Set Size",
                "Num Segments",
                "Measure Method",
                "Pixel Norm Rate",
                "Food Sides",
                "Training Time (s)",
                "Elapsed Time (s)",
                "k",
                "Good Matches",
                "Bad Matches",
                "Accuracy",
            ])

        # Unpack the single test_summary
        (
            treesCount,
            leavesCount,
            training_set_size,
            test_set_size,
            num_segments,
            measure_method,
            pixel_norm_rate,
            flood_sides,
            training_time,
            k_summaries,
        ) = test_summary

        for k_summary in k_summaries:
            elapsed_time, k, good, bad, accuracy, matrix = k_summary

            # Process flood_sides into a string
            flood_sides_string = ""
            if flood_sides[0] == "1":
                flood_sides_string += "left, "
            if flood_sides[1] == "1":
                flood_sides_string += "right, "
            if flood_sides[2] == "1":
                flood_sides_string += "top, "
            if flood_sides[3] == "1":
                flood_sides_string += "bottom, "
            # Append original flood_sides (assuming it's a string or can be concatenated)
            flood_sides_string += str(flood_sides)
            # Remove any trailing ", " from the directional labels
            flood_sides_string = flood_sides_string.rstrip(', ')

            # Write to params CSV
            params_writer.writerow([
                test_id,
                treesCount,
                leavesCount,
                training_set_size,
                test_set_size,
                num_segments,
                measure_method,
                pixel_norm_rate,
                flood_sides_string,
                round(training_time, 2),
                round(elapsed_time, 2),
                k,
                good,
                bad,
                round(accuracy, 4),
            ])

            # Write to matrices CSV
            for actual_class, predictions in enumerate(matrix):
                matrices_writer.writerow([
                    test_id,
                    actual_class,
                    *predictions
                ])
            matrices_writer.writerow([])

            test_id += 1  # Increment for each k_summary

    print(f"Appended data to {params_filename} and {matrices_filename}")

def test(date, mode="ann"):
    

    testSummaries = []
    testCases =  ANNtestCases
    for index, testCase in enumerate(testCases):
        print(f"Testing case no.{index}")
        dataset = "train"
        pixels, labels = load_data(dataset)

        treesCount = testCase[0]
        leavesCount = testCase[1]
        trainingSetSize = testCase[2]
        numSegments = testCase[3]
        pixelNormalizationRate = testCase[4]
        floodSides = testCase[5]

        pixels = pixels[:trainingSetSize]
        labels = labels[:trainingSetSize]


        start_time = time.perf_counter()
        train_vectors, train_labels = generate_training_vectors(pixels, labels, trainingSetSize, numSegments, pixelNormalizationRate, floodSides=floodSides)

        print("Generating Forest")
        ann = Ann(train_vectors, train_labels, treesCount, leavesCount, 0.95)
        print("Forest Generated")

        end_time = time.perf_counter()
        training_time = end_time - start_time
        # training end

        pixels, labels = load_data("test")


        #query start

        kSummary = test_annoy_singular(ann, pixels,labels,trainingSetSize,numSegments, pixelNormalizationRate, floodSides=floodSides)

        print("Saving results")
        generate_csv_from_test_summary(
                [
                    treesCount,
                    leavesCount,
                    trainingSetSize,
                    10000,
                    numSegments,
                    "annoy",
                    pixelNormalizationRate,
                    floodSides,
                    training_time,
                    kSummary,
                ],
                date
        )

      
        print(f"Test no.{index} Completed")
    print("All test completed")
def test_annoy_singular(ann, pixels, labels, trainingSetSize, numSegments, pixelNormalizationRate, floodSides="1111"):
    summaryMatrix = np.zeros((10, 10), dtype=int)
    good_match = 0
    bad_match = 0
    start_time = time.perf_counter()
    for i in range(len(pixels)):

        binarized_data, label = get_data(pixels, labels, i, pixelNormalizationRate)
        vec = create_vector_for_one_number(binarized_data, label, numSegments, floodSides=floodSides)

        # separate label
        label = vec[0]
        # separate vector
        vec = vec[1:]

        # predict value
        val = ann.predict_label(vec, 10)

        if val == label:
            good_match += 1
        else:
            bad_match += 1
        summaryMatrix[int(label)][int(val)] += 1
    end_time = time.perf_counter()
    print(f"Test Completed: Good Matches: {good_match}, Bad matches {bad_match},  % {good_match / (good_match + bad_match)}")
    return   [[
                    end_time-start_time,
                    'n/a',
                    good_match,
                    bad_match,
                    good_match / (good_match + bad_match),
                    summaryMatrix,
                ]]