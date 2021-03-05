#!/usr/bin/env python3
import argparse
import lzma
import pickle
import os
import urllib.request
import sys
import zipfile
import sklearn.model_selection
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import sklearn.pipeline


class Dataset:
    CLASSES = ["sitting", "sittingdown", "standing", "standingup", "walking"]

    def __init__(self,
                 name="human_activity_recognition.train.csv.xz",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2021/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset and if it contains column "class", split it to `targets`.
        self.data = pd.read_csv(name)
        if "class" in self.data:
            self.target = np.array([Dataset.CLASSES.index(target) for target in self.data["class"]], np.int32)
            self.data = self.data.drop("class", axis=1)

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Run prediction on given data")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="human_activity_recognition.model", type=str, help="Model path")
parser.add_argument("--test_size", default=0.2, type=lambda x: int(x) if x.isdigit() else float(x),
                    help="Test set size")
def main(args):
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()
        print(train.data)
        print(train.target)

        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(train.data, train.target,
                                                                                    test_size=args.test_size,
                                                                                    random_state=args.seed)

        # TODO: Train a model on the given dataset and store it in `model`.
        pipeline = sklearn.pipeline.Pipeline(steps=[("rnd", RandomForestClassifier())])

        params = {
        "rnd__criterion": ["gini", "entropy"],
        "rnd__min_samples_split" : [2,3,4],
        "rnd__min_samples_leaf" : [1,2],
        "rnd__max_features" : ["auto", "sqrt", "log2"]
        }

        model = GridSearchCV(pipeline, params, cv=5)

        model.fit(X_train, y_train)
        test_predicted = model.predict(X_test)

        test_accuracy = sum(test_predicted == y_test) / len(y_test)
        print(test_accuracy)
        print(model.best_params_)


        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # TODO: Generate `predictions` with the test set predictions, either
        # as a Python list of a NumPy array.
        predictions = model.predict(test.data)

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)