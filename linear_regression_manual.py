#!/usr/bin/env python3

import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.9, type=lambda x: int(x) if x.isdigit() else float(x),
                    help="Test set size")


# a507688b-17c7-11e8-9de3-00505601122b.
# If you add more arguments, ReCodEx will keep them with your default values.

def main(args):
    # Load Boston housing dataset
    dataset = sklearn.datasets.load_boston()

    # The input data are in dataset.data, targets are in dataset.target.

    # If you want to learn about the dataset, uncomment the following line.
    # print(dataset.DESCR)

    # TODO: Append a new feature to all input data, with value "1"

    dataset_with_intercept = np.append(dataset.data, np.repeat([[1]], dataset.data.shape[0], axis=0), axis=1)
    dataset_target = np.reshape(dataset.target, [dataset.target.shape[0], 1])

    # TODO: Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(dataset_with_intercept, dataset_target,
                                                                                test_size=args.test_size,
                                                                                random_state=args.seed)

    # TODO: Solve the linear regression using the algorithm from the lecture,
    # explicitly computing the matrix inverse (using `np.linalg.inv`).

    weights = np.matmul(np.matmul(np.linalg.inv(np.matmul(x_train.transpose(), x_train)), x_train.transpose()), y_train)

    # TODO: Predict target values on the test set

    y_hat = np.matmul(x_test, weights)

    # TODO: Compute root mean square error on the test set predictions

    rmse = np.sqrt((np.linalg.norm(y_hat - y_test, ord=2) ** 2) / y_test.shape[0])

    return rmse


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    rmse = main(args)
    print("{:.2f}".format(rmse))
