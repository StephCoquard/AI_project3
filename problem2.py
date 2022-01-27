import sys
import numpy as np
from statistics import mean
from statistics import stdev

iterations = 100
alphas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 3]


def normalize(matrix):
    normalized_matrix = np.copy(matrix)
    nb_of_features = normalized_matrix.shape[1]
    for feature_index in range(nb_of_features):
        mean_value = mean(normalized_matrix[:, feature_index])
        st_dev_value = stdev(normalized_matrix[:, feature_index])
        normalized_matrix[:, feature_index] = (normalized_matrix[:, feature_index] - mean_value) / st_dev_value

    return normalized_matrix


def gradient_descent(matrix, labels, alpha):
    theta = np.zeros(matrix.shape[1])
    n = matrix.shape[0]

    for i in range(iterations):
        theta -= alpha * (1/n) * np.dot(matrix.T, (np.dot(matrix, theta) - labels))
    return alpha, iterations, theta[0], theta[1], theta[2]


def main():
    input_file = sys.argv[1].lower()
    output_file = sys.argv[2].lower()
    raw_data = np.loadtxt(input_file, delimiter=',')

    normalized_matrix = normalize(raw_data[:, :-1])
    b = np.ones(normalized_matrix.shape[0])
    matrix = np.column_stack((b, normalized_matrix))
    labels = raw_data[:, -1]

    fo = open(output_file, 'w')
    for alpha in alphas:
        result = gradient_descent(matrix, labels, alpha)
        fo.write("%s, %d, %s, %s, %s\n" % result[:])


if __name__ == "__main__":
    main()
