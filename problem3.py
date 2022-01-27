import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def svm_with_linear_kernel(x_train, x_test, y_train, y_test, fo):
    params = {
        'C': [0.1, 0.5, 1, 5, 10, 50, 100],
        'degree': [4, 5, 6],
        'gamma': [0.1, 1],
        'kernel': ['linear']
    }

    grid_thing = GridSearchCV(SVC(), params, n_jobs=1)
    grid_thing.fit(x_train, y_train)
    fo.write("%s, %0.2f, %0.2f\n" % ('svm_linear', grid_thing.best_score_, grid_thing.score(x_test, y_test)))


def svm_with_polynomial_kernel(x_train, x_test, y_train, y_test, fo):
    params = {
        'C': [0.1, 1, 3],
        'degree': [4, 5, 6],
        'gamma': [0.1, 1],
        'kernel': ['poly']
    }

    grid_thing = GridSearchCV(SVC(), params, n_jobs=1)
    grid_thing.fit(x_train, y_train)
    fo.write("%s, %0.2f, %0.2f\n" % ('svm_polynomial', grid_thing.best_score_, grid_thing.score(x_test, y_test)))


def svm_with_rbf_kernel(x_train, x_test, y_train, y_test, fo):
    params = {
        'C': [0.1, 0.5, 1, 5, 10, 50, 100],
        'gamma': [0.1, 0.5, 1, 3, 6, 10],
        'kernel': ['rbf']
    }

    grid_thing = GridSearchCV(SVC(), params, n_jobs=1)
    grid_thing.fit(x_train, y_train)
    fo.write("%s, %0.2f, %0.2f\n" % ('svm_rbf', grid_thing.best_score_, grid_thing.score(x_test, y_test)))


def logistic_regression(x_train, x_test, y_train, y_test, fo):
    params = {
        'C': [0.1, 0.5, 1, 5, 10, 50, 100],
        'solver': ['liblinear']
    }

    grid_thing = GridSearchCV(LogisticRegression(), params, n_jobs=1)
    grid_thing.fit(x_train, y_train)
    fo.write("%s, %0.2f, %0.2f\n" % ('logistic', grid_thing.best_score_, grid_thing.score(x_test, y_test)))


def k_nearest_neighbors(x_train, x_test, y_train, y_test, fo):
    params = {
        'n_neighbors': range(1, 51),
        'leaf_size': range(5, 65, 5),
        'algorithm': ['auto']
    }

    grid_thing = GridSearchCV(KNeighborsClassifier(), params, n_jobs=1)
    grid_thing.fit(x_train, y_train)
    fo.write("%s, %0.2f, %0.2f\n" % ('knn', grid_thing.best_score_, grid_thing.score(x_test, y_test)))


def decision_trees(x_train, x_test, y_train, y_test, fo):
    params = {
        'max_depth': range(1, 51),
        'min_samples_split': range(2, 11)
    }

    grid_thing = GridSearchCV(DecisionTreeClassifier(), params, n_jobs=1)
    grid_thing.fit(x_train, y_train)
    fo.write("%s, %0.2f, %0.2f\n" % ('decision_tree', grid_thing.best_score_, grid_thing.score(x_test, y_test)))


def random_forest(x_train, x_test, y_train, y_test, fo):
    params = {
        'max_depth': range(1, 51),
        'min_samples_split': range(2, 11)
    }

    grid_thing = GridSearchCV(RandomForestClassifier(), params, n_jobs=1)
    grid_thing.fit(x_train, y_train)
    fo.write("%s, %0.2f, %0.2f\n" % ('random_forest', grid_thing.best_score_, grid_thing.score(x_test, y_test)))


def main():
    input_file = sys.argv[1].lower()
    output_file = sys.argv[2].lower()
    fo = open(output_file, 'w')

    raw_data = np.loadtxt(input_file, delimiter=',', skiprows=1)
    x = raw_data[:, :-1]
    y = raw_data[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)

    svm_with_linear_kernel(x_train, x_test, y_train, y_test, fo)
    svm_with_polynomial_kernel(x_train, x_test, y_train, y_test, fo)
    svm_with_rbf_kernel(x_train, x_test, y_train, y_test, fo)
    logistic_regression(x_train, x_test, y_train, y_test, fo)
    k_nearest_neighbors(x_train, x_test, y_train, y_test, fo)
    decision_trees(x_train, x_test, y_train, y_test, fo)
    random_forest(x_train, x_test, y_train, y_test, fo)


if __name__ == "__main__":
    main()
