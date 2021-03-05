#!/usr/bin/env python3
import argparse
import lzma
import os
import pickle
import urllib.request

import numpy as np
import sklearn
import sklearn.compose
import sklearn.datasets
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.linear_model
import sklearn.metrics


class Dataset:
    """Rental Dataset.
    The dataset instances consist of the following 12 features:
    - season (1: winter, 2: sprint, 3: summer, 4: autumn)
    - year (0: 2011, 1: 2012)
    - month (1-12)
    - hour (0-23)
    - holiday (binary indicator)
    - day of week (0: Sun, 1: Mon, ..., 6: Sat)
    - working day (binary indicator; a day is neither weekend nor holiday)
    - weather (1: clear, 2: mist, 3: light rain, 4: heavy rain)
    - temperature (normalized so that -8 Celsius is 0 and 39 Celsius is 1)
    - feeling temperature (normalized so that -16 Celsius is 0 and 50 Celsius is 1)
    - relative humidity (0-1 range)
    - windspeed (normalized to 0-1 range)
    The target variable is the number of rentals in the given hour.
    """

    def __init__(self,
                 name="rental_competition.train.npz",
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
parser.add_argument("--model_path", default="rental_competition.model", type=str, help="Model path")


def AnotherTransform(data):
    enc = sklearn.preprocessing.OneHotEncoder(handle_unknown="ignore", sparse=False)
    scaler = sklearn.preprocessing.StandardScaler()
    categorical_data = data[:, np.all(data.astype(int) == data, axis=0)]
    continuous_data = data[:, ~np.all(data.astype(int) == data, axis=0)]

    if np.sum(np.all(data.astype(int) == data, axis=0)) > 0:
        categorical_data_encoded = enc.fit_transform(categorical_data)
        data_transformed = categorical_data_encoded
    if np.sum(~np.all(data.astype(int) == data, axis=0)) > 0:
        continuous_data_normalized = scaler.fit_transform(continuous_data)
        data_transformed = continuous_data_normalized
    if np.sum(np.all(data.astype(int) == data, axis=0)) > 0 and np.sum(
            ~np.all(data.astype(int) == data, axis=0)) > 0:
        data_transformed = np.append(categorical_data_encoded, continuous_data_normalized, axis=1)
    # poly = sklearn.preprocessing.PolynomialFeatures(2, include_bias=False)
    # data_transformed = poly.fit_transform(data_transformed)
    return data_transformed


def main(args):
    if args.predict is None:
        # We are training a model.

        np.random.seed(args.seed)
        train = Dataset()
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(AnotherTransform(train.data),
                                                                                    train.target, test_size=10,
                                                                                    random_state=args.seed)

        # TODO: Train a model on the given dataset and store it in `model`.
        categoical = [True, True, True, True, True, True, True, True, False, False, False, False]
        continuous = [not a for a in categoical]

        # pipe = sklearn.pipeline.Pipeline([("column_transf", sklearn.compose.ColumnTransformer(
        #     [("one_hot", sklearn.preprocessing.OneHotEncoder(handle_unknown="ignore", sparse=False), categoical),
        #      ("scale", sklearn.preprocessing.StandardScaler(), continuous)])),
        #                                   ("poly", sklearn.preprocessing.PolynomialFeatures(2, include_bias=False))])
        print(x_train)
        colum = sklearn.compose.ColumnTransformer(
            [("one_hot", sklearn.preprocessing.OneHotEncoder(handle_unknown="ignore", sparse=False), categoical),
             ("scale", sklearn.preprocessing.StandardScaler(), continuous)])

        print(colum.fit_transform(x_train))
        # model = sklearn.linear_model.LinearRegression()
        # model.fit(colum.fit_transform(x_train), y_train)
        # y_hat = model.predict(colum.transform(x_test))
        # rmse = np.sqrt(sklearn.metrics.mean_squared_error(y_hat, y_test))
        # print(rmse)

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions, as either a Python list or a NumPy array.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # TODO: Generate `predictions` with the test set predictions.

        predictions = model.predict(AnotherTransform(test.data))

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
