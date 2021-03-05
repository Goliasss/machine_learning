#!/usr/bin/env python3
import argparse
import lzma
import os
import pickle
import urllib.request

import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.compose


class Dataset:
    """Thyroid Dataset.
    The dataset contains real medical data related to thyroid gland function,
    classified either as normal or irregular (i.e., some thyroid disease).
    The data consists of the following features in this order:
    - 15 binary features
    - 6 real-valued features
    The target variable is binary, with 1 denoting a thyroid disease and
    0 normal function.
    """

    def __init__(self,
                 name="thyroid_competition.train.npz",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2021/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name))
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset and return the data and targets.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value)


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Run prediction on given data")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="thyroid_competition.model", type=str, help="Model path")
parser.add_argument("--test_size", default=0.2, type=lambda x: int(x) if x.isdigit() else float(x),
                    help="Test set size")


def main(args):
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        data = np.append(train.data, np.repeat([[1]], train.data.shape[0], axis=0), axis=1)

        train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(data, train.target,
                                                                                                    test_size=args.test_size,
                                                                                                    random_state=args.seed)
        int_columns = np.all(train_data.astype(int) == train_data, axis=0)
        pipeline = sklearn.pipeline.Pipeline([
            ("preprocess", sklearn.compose.ColumnTransformer([
                ("onehot", sklearn.preprocessing.OneHotEncoder(handle_unknown="ignore"), int_columns),
                ("scaler", sklearn.preprocessing.StandardScaler(), ~int_columns),
            ])),
            ("poly", sklearn.preprocessing.PolynomialFeatures(include_bias=False)),
            ('log', sklearn.linear_model.LogisticRegression(random_state=args.seed))
        ])

        parameteres = {'poly__degree': [3], 'log__C': [2.5,2.75,3,3.25,3.5],
                       'log__solver': ['lbfgs']}

        model = sklearn.model_selection.GridSearchCV(pipeline, param_grid=parameteres, cv=5)
        model.fit(train_data, train_target)
        test_predicted = model.predict(test_data)

        test_accuracy = sum(test_predicted == test_target) / len(test_target)
        print(test_accuracy)
        print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        print(model.best_params_)
        # TODO: Train a model on the given dataset and store it in `model`.

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions, as either a Python list or a NumPy array.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # TODO: Generate `predictions` with the test set predictions.

        test = np.append(test.data, np.repeat([[1]], test.data.shape[0], axis=0), axis=1)

        predictions = model.predict(test)

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
