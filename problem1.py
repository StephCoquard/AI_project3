import sys
import numpy as np


def run(data, labels):

    correct = False

    weights_history = []

    weights = np.zeros(3)

    while not correct:
        weights, correct = iterate(data, labels, weights)
        weights_history.append(weights.copy())

    return weights_history


def iterate(data, labels, weights):
    for (row, label) in zip(data, labels):
        prediction = predict(row, weights)
        if label < prediction:
            weights -= row
        elif label > prediction:
            weights += row

    correct = test(data, labels, weights)
    return weights, correct


def predict(row, weights):
    arrays_product = np.dot(row, weights)
    if arrays_product > 0:
        return 1
    else:
        return -1


def test(data, labels, weights):
    correct = True
    for (row, label) in zip(data, labels):
        prediction = predict(row, weights)
        if label != prediction:
            correct = False
            break
    return correct


def main():
    input_file = sys.argv[1].lower()
    output_file = sys.argv[2].lower()
    raw_data = np.loadtxt(input_file, delimiter=',')
    rows_nb = raw_data.shape[0]
    data = raw_data[:, [0, 1]]
    ones_array = np.ones(rows_nb)
    ones_array.shape = (rows_nb, 1)
    labels = raw_data[:, [-1]].flatten()
    data = np.hstack((data, ones_array))

    weights_history = run(data, labels)

    fo = open(output_file, 'w')
    for weights in weights_history:
        fo.write("%d, %d, %d\n" % (weights[0], weights[1], weights[2]))


if __name__ == "__main__":
    main()
