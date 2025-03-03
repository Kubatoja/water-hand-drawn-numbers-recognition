from Data.data import load_data, binarize_data, load_vectors
from VectorGeneration.vectors import *
from KNN.knn import knn
import csv


# testcases
# 1: Distancemode
# 2: knn range start
# 3: knn range end
# 4: Training set size (1-9999)
# 5: num segments
testCases = [
    [1, 1,7, 8572, 7, 0.15],
    [4, 1,7, 8572, 7, 0.15],
    [2, 1,7, 8572, 7, 0.15],   
    [1, 1,7, 8572, 7, 0.25],
    [4, 1,7, 8572, 7, 0.25],
    [2, 1,7, 8572, 7, 0.25],
    [1, 1,7, 8572, 7, 0.35],
    [4, 1,7, 8572, 7, 0.35],
    [2, 1,7, 8572, 7, 0.35],
    [1, 1,7, 8572, 7, 0.45],
    [4, 1,7, 8572, 7, 0.45],
    [2, 1,7, 8572, 7, 0.45],
    [1, 1,7, 8572, 7, 0.55],
    [4, 1,7, 8572, 7, 0.55],
    [2, 1,7, 8572, 7, 0.55],
    [1, 1,7, 8572, 7, 0.65],
    [4, 1,7, 8572, 7, 0.65],
    [2, 1,7, 8572, 7, 0.65],
    [1, 1,7, 8572, 7, 0.75],
    [4, 1,7, 8572, 7, 0.75],
    [2, 1,7, 8572, 7, 0.75],
    [1, 1,7, 8572, 7, 0.85],
    [4, 1,7, 8572, 7, 0.85],
    [2, 1,7, 8572, 7, 0.85]
    ]


# load all data
pixels = None
labels = None
train_labels = None
train_vectors = None


def train(trainingSetSize, numSegments, pixelNormalizationRate):
    global train_labels, train_vectors
    generate_vectors_for_n(trainingSetSize, numSegments, pixels, labels, pixelNormalizationRate)
    train_labels, train_vectors = load_vectors()
    print("generated")


def generate_csv_from_test_summaries(testSummaries, filename="test_results.csv"):
    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        # Write header row
        writer.writerow(
            [
                "Training Set Size",
                "Testing Set Size",
                "Num Segments",
                "Distance Mode",
                "k",
                "Good Matches",
                "Bad Matches",
                "Accuracy",
                "Confusion Matrix",
            ]
        )

        for test_summary in testSummaries:
            (
                training_set_size,
                testing_set_size,
                num_segments,
                distance_mode,
                k_summaries,
            ) = test_summary

            for k_summary in k_summaries:
                k, good_matches, bad_matches, accuracy, summary_matrix = k_summary

                # Write basic summary row
                writer.writerow(
                    [
                        training_set_size,
                        testing_set_size,
                        num_segments,
                        distance_mode,
                        k,
                        good_matches,
                        bad_matches,
                        accuracy,
                        "Confusion Matrix:",
                    ]
                )

                # Prepare header row for confusion matrix
                header_row = [
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "Predicted/Actual",
                ] + list(range(10))
                writer.writerow(header_row)

                # Write confusion matrix rows with row labels
                for row_index, row in enumerate(summary_matrix):
                    matrix_row = [
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        str(row_index),
                    ] + list(row)
                    writer.writerow(matrix_row)

                writer.writerow([])  # Blank row to separate the matrices.


def test():
    global pixels, labels, train_labels, train_vectors
    pixels, labels = load_data("test")

    testSummaries = []
    for testCase in testCases:
        distanceMode = testCase[0]
        knnRangeStart = testCase[1]
        knnRangeEnd = testCase[2]
        trainingSetSize = testCase[3]
        numSegments = testCase[4]
        pixelNormalizationRate = testCase[5]
        train(trainingSetSize, numSegments, pixelNormalizationRate)

        # test for different k values
        kSummary = []

        for k in range(knnRangeStart, knnRangeEnd + 1):

            summaryMatrix = np.zeros((10, 10), dtype=int)
            good_match = 0
            bad_match = 0

            # iterate throu untrained data
            for i in range(trainingSetSize, 10000 - 1):

                binarized_data, label = get_data(pixels, labels, i, pixelNormalizationRate)
                vec = create_vector_for_one_number(binarized_data, label, numSegments)

                # separate label
                label = vec[0]
                # separate vector
                vec = vec[1:]

                # predict value
                val = knn(vec, train_labels, train_vectors, k, distanceMode)

                if val == label:
                    good_match += 1
                else:
                    bad_match += 1
                summaryMatrix[int(label)][int(val)] += 1

            kSummary.append(
                [
                    k,
                    good_match,
                    bad_match,
                    good_match / (good_match + bad_match),
                    summaryMatrix,
                ]
            )
            print(k, good_match, bad_match, good_match / (good_match + bad_match))

        testSummaries.append(
            [
                trainingSetSize,
                10000 - trainingSetSize,
                numSegments,
                distanceMode,
                kSummary,
            ]
        )
        print("test completed")

    generate_csv_from_test_summaries(testSummaries)
    print("summary generated")
