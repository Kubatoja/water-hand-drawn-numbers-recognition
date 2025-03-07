from Data.data import load_data, binarize_data, load_vectors
from VectorGeneration.vectors import *
from VectorSearch.knn import *
from VectorSearch.annoy import *
import csv


# testcases
# 1: Distancemode
# 2: knn range start
# 3: knn range end
# 4: Training set size (1-9999)
# 5: num segments
# 6: pixelNormalizationRate
# 7: floodSides(Left, Right, Top, Bottom) STRING!!!
KNNtestCases = [
    [3, 5,7, 8572, 5, 0.314, "1111"]
    ]

# testcases
# 1: Number of trees
# 2: Number of leaves
# 3: Training set size (1-9999)
# 4: num segments
# 5: pixelNormalizationRate
# 6: floodSides(Left, Right, Top, Bottom) STRING!!!

ANNtestCases = [
    [2, 32, 8572, 5, 0.34, "1111"]
    ]


def generate_training_vectors(pixels, labels, trainingSetSize, numSegments, pixelNormalizationRate, floodSides="1111"):
    print("Generating Vectors")
    generate_vectors_for_n(trainingSetSize, numSegments, pixels, labels, pixelNormalizationRate, floodSides=floodSides)
    print(f"Generated vectors for {trainingSetSize} numbers")

    print("Loading Vectors")
    train_vectors, train_labels = load_vectors()
    print(f"Loaded {len(train_labels)} Vectors")
    return train_vectors, train_labels




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
    print(f"Summary generated in {filename}")

def test(mode="ann"):
    pixels, labels = load_data("test")

    testSummaries = []
    testCases = KNNtestCases if mode == "knn" else ANNtestCases
    for index, testCase in enumerate(testCases):
        print(f"Testing case no.{index}")
        if(mode == "knn"):
            distanceMode = testCase[0]
            knnRangeStart = testCase[1]
            knnRangeEnd = testCase[2]
            trainingSetSize = testCase[3]
            numSegments = testCase[4]

            pixelNormalizationRate = testCase[5]
            floodSides = testCase[6]
        elif(mode == "ann"):
            treesCount = testCase[0]
            leavesCount = testCase[1]
            trainingSetSize = testCase[2]
            numSegments = testCase[3]
            pixelNormalizationRate = testCase[4]
            floodSides = testCase[5]



        train_vectors, train_labels = generate_training_vectors(pixels, labels, trainingSetSize, numSegments, pixelNormalizationRate, floodSides=floodSides)

        if(mode == "ann"):
            print("Generating Forest")
            forest = build_forest(train_vectors, train_labels, treesCount, leavesCount, 0.95)
            print("Forest Generated")
            kSummary = test_annoy_singular(pixels,labels,forest,trainingSetSize,numSegments, pixelNormalizationRate, floodSides=floodSides)

        elif(mode == "knn"):
            # test for different k values
            kSummary = test_knn_range(pixels, labels, train_vectors, train_labels, knnRangeStart, knnRangeEnd, trainingSetSize, numSegments, distanceMode, pixelNormalizationRate, floodSides=floodSides)


        if(mode == "knn"):
            testSummaries.append(
                [
                    trainingSetSize,
                    10000 - trainingSetSize,
                    numSegments,
                    distanceMode,
                    kSummary,
                ]
            )
        elif(mode == "ann"):
            testSummaries.append(
                [
                    trainingSetSize,
                    10000 - trainingSetSize,
                    numSegments,
                    "annoy",
                    kSummary,
                ]
            )

      
        print(f"Test no.{index} Completed")

    print("All test completed, generating summary")
    generate_csv_from_test_summaries(testSummaries)



def test_knn_range(pixels, labels, train_vectors, train_labels, knnRangeStart, knnRangeEnd, trainingSetSize, numSegments, distanceMode, pixelNormalizationRate):
    kSummary = []
    for k in range(knnRangeStart, knnRangeEnd + 1):
            print(f"Testing for k = {k}")
            good_match, bad_match, summaryMatrix = test_knn_singular(pixels, labels, train_vectors, train_labels, k, trainingSetSize, numSegments, distanceMode, pixelNormalizationRate)
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
    return kSummary
def test_knn_singular(pixels, labels, train_vectors, train_labels, k, trainingSetSize, numSegments, distanceMode, pixelNormalizationRate, floodSides="1111"):
        summaryMatrix = np.zeros((10, 10), dtype=int)
        good_match = 0
        bad_match = 0

        for i in range(trainingSetSize, 10000 - 1):

            binarized_data, label = get_data(pixels, labels, i, pixelNormalizationRate)
            vec = create_vector_for_one_number(binarized_data, label, numSegments, floodSides=floodSides)

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

        return good_match, bad_match, summaryMatrix


def test_annoy_singular(pixels, labels, forest, trainingSetSize, numSegments, pixelNormalizationRate, floodSides="1111"):
    summaryMatrix = np.zeros((10, 10), dtype=int)
    good_match = 0
    bad_match = 0

    for i in range(trainingSetSize, 10000 - 1):

        binarized_data, label = get_data(pixels, labels, i, pixelNormalizationRate)
        vec = create_vector_for_one_number(binarized_data, label, numSegments, floodSides=floodSides)

        # separate label
        label = vec[0]
        # separate vector
        vec = vec[1:]

        # predict value
        val = approximate_label(forest, vec, 10)

        if val == label:
            good_match += 1
        else:
            bad_match += 1
        summaryMatrix[int(label)][int(val)] += 1
    print(good_match, bad_match, good_match / (good_match + bad_match))
    return   [[      0,
                    good_match,
                    bad_match,
                    good_match / (good_match + bad_match),
                    summaryMatrix,
                ]]