#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
from scipy import stats

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--C", default=1, type=float, help="Inverse regularization strength")
parser.add_argument("--classes", default=3, type=int, help="Number of classes")
parser.add_argument("--kernel", default="poly", type=str, help="Kernel type [poly|rbf]")
parser.add_argument("--kernel_degree", default=1, type=int, help="Degree for poly kernel")
parser.add_argument("--kernel_gamma", default=1.0, type=float, help="Gamma for poly and rbf kernel")
parser.add_argument("--max_iterations", default=1000, type=int, help="Maximum number of iterations to perform")
parser.add_argument("--max_passes_without_as_changing", default=10, type=int, help="Number of passes without changes to stop after")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
parser.add_argument("--tolerance", default=1e-4, type=float, help="Default tolerance for KKT conditions")
# If you add more arguments, ReCodEx will keep them with your default values.
def libsvm(args, train_data, train_target):
    import sklearn.svm
    clf = sklearn.svm.SVC(C=args.C, kernel=args.kernel, degree=args.kernel_degree, gamma=args.kernel_gamma, tol=args.tolerance, coef0=1)
    clf.fit(train_data, train_target)
    return clf

def kernel(args, x, y):
    # TODO: As in `kernel_linear_regression`, We consider the following `args.kernel`s:
    # - "poly": K(x, y; degree, gamma) = (gamma * x^T y + 1) ^ degree
    # - "rbf": K(x, y; gamma) = exp^{- gamma * ||x - y||^2}
    if args.kernel == "poly":
        return (args.kernel_gamma * x @ y + 1) ** args.kernel_degree
    if args.kernel == "rbf":
        return np.exp(-args.kernel_gamma * np.sum((x - y) * (x - y), axis=-1))
def smo(args, train_data, train_target):
    # Create initial weights
    a, b = np.zeros(len(train_data)), 0
    generator = np.random.RandomState(args.seed)

    K = np.array([[kernel(args, x, y) for x in train_data] for y in train_data])

    passes_without_as_changing = 0
    train_accs, test_accs = [], []
    for _ in range(args.max_iterations):
        as_changed = 0
        # Iterate through the data
        for i, j in enumerate(generator.randint(len(a) - 1, size=len(a))):
            # We want j != i, so we "skip" over the value of i
            j = j + (j >= i)

            # TODO: Check that a[i] fulfils the KKT conditions, using `args.tolerance` during comparisons.
            Ei = (a * train_target) @ K[i] + b - train_target[i]
            if not (a[i] < args.C - args.tolerance and train_target[i] * Ei < -args.tolerance) \
                    and not (a[i] > args.tolerance and train_target[i] * Ei > args.tolerance):
                continue

            # If the conditions do not hold, then
            # - compute the updated unclipped a_j^new.
            #
            #   If the second derivative of the loss with respect to a[j]
            #   is > -`args.tolerance`, do not update a[j] and continue
            #   with next i.
            eta = 2 * K[i, j] - K[i, i] - K[j, j]
            if eta > -args.tolerance:
                continue

            Ej = (a * train_target) @ K[j] + b - train_target[j]
            new_aj = a[j] - train_target[j] * (Ei - Ej) / eta

            if train_target[i] == train_target[j]:
                L, H = max(0, a[i] + a[j] - args.C), min(args.C, a[i] + a[j])
            else:
                L, H = max(0, a[j] - a[i]), min(args.C, args.C + a[j] - a[i])

            new_aj = np.clip(new_aj, L, H)
            if abs(new_aj - a[j]) < args.tolerance:
                continue

            # - update a[j] to a_j^new, and compute the updated a[i] and b.
            #
            #   During the update of b, compare the a[i] and a[j] to zero by
            #   `> args.tolerance` and to C using `< args.C - args.tolerance`.
            new_ai = a[i] - train_target[i] * train_target[j] * (new_aj - a[j])

            bi = b - Ei - train_target[i] * (new_ai - a[i]) * K[i, i] - train_target[j] * (new_aj - a[j]) * K[j, i]
            bj = b - Ej - train_target[i] * (new_ai - a[i]) * K[i, j] - train_target[j] * (new_aj - a[j]) * K[j, j]
            a[i], a[j] = new_ai, new_aj
            if args.tolerance < a[i] < args.C - args.tolerance:
                b = bi
            elif args.tolerance < a[j] < args.C - args.tolerance:
                b = bj
            else:
                b = (bi + bj) / 2

            # - increase `as_changed`
            as_changed += 1

        # Stop training if max_passes_without_as_changing passes were reached
        passes_without_as_changing = 0 if as_changed else passes_without_as_changing + 1
        if passes_without_as_changing >= args.max_passes_without_as_changing:
            break

    support_vectors = train_data[a > args.tolerance]
    support_vector_weights = (a * train_target)[a > args.tolerance]

    return support_vectors, support_vector_weights, b
def main(args):
    # Use the digits dataset.
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)
    data = sklearn.preprocessing.MinMaxScaler().fit_transform(data)

    # Split the dataset into a train set and a test set.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # TODO: Using One-vs-One scheme, train (K \binom 2) classifiers, one for every
    # pair of classes $i < j$, using the `smo` method.
    #
    # When training a classifier for classes $i < j$:
    # - keep only the training data of these classes, in the same order
    #   as in the input dataset;
    # - use targets 1 for the class $i$ and -1 for the class $j$.

    def kernel(x, z):
        if args.kernel == "poly":
            return (args.kernel_gamma * np.dot(x, z) + 1) ** args.kernel_degree
        elif args.kernel == "rbf":
            return np.exp(-args.kernel_gamma * (np.linalg.norm(x - z)) ** 2)

    def y(x):
        return np.sum(
            [support_vector_weights[i] * kernel(x, support_vectors[i]) for i in range(len(support_vectors))]) + bias

    def process_data(data, target, i, j):
        one_train_target = target[np.logical_or(target == i, target == j)]
        for k in range(len(one_train_target)):
            if one_train_target[k] == i:
                one_train_target[k] = 1
            else:
                one_train_target[k] = -1
        one_train_data = data[np.logical_or(target == i, target == j)]
        return one_train_data, one_train_target

    voting = [list([]) for _ in range(len(test_data))]

    for i in range(args.classes):
        for j in range(i+1, args.classes):
            one_train_data, one_train_target = process_data(train_data, train_target, i, j)

            support_vectors, support_vector_weights, bias = smo(args, one_train_data, one_train_target)

            test_prediction = np.array([np.sign(y(test_data[i])) for i in range(len(test_data))])

            for k in range(len(test_data)):
                if test_prediction[k] == 1:
                    voting[k] = np.append(voting[k], i)
                    # list(voting[k]).append(i)
                else:
                    # list(voting[k]).append(j)
                    voting[k] = np.append(voting[k], j)
    # TODO: Classify the test set by majority voting of all the trained classifiers,
    # using the lowest class index in the case of ties.
    #
    # Finally, compute the test set prediction accuracy.
    test_prediction = []
    for i in range(len(test_data)):
        test_prediction.append(int(stats.mode(voting[i])[0]))
        # test_prediction = np.append(test_prediction, int(stats.mode(voting[i])[0]))
    test_accuracy = sum(test_prediction == test_target) / len(test_target)

    return test_accuracy

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    accuracy = main(args)
    print("Test set accuracy: {:.2f}%".format(100 * accuracy))
