#!/usr/bin/env python3
import argparse
import lzma
import pickle
import os

import sklearn.pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
import numpy as np

class Dataset:
    CLASSES = ["ARA", "DEU", "FRA", "HIN", "ITA", "JPN", "KOR", "SPA", "TEL", "TUR", "ZHO"]

    def __init__(self, name):
        if not os.path.exists(name):
            raise RuntimeError("The {} was not found, please download it from ReCodEx".format(name))

        # Load the dataset and split it into `data` and `target`.
        self.data, self.prompts, self.levels, self.target = [], [], [], []
        with open(name, "r", encoding="utf-8") as dataset_file:
            for line in dataset_file:
                target, prompt, level, text = line.rstrip("\n").split("\t")
                self.data.append(text)
                self.prompts.append(prompt)
                self.levels.append(level)
                self.target.append(-1 if not target else self.CLASSES.index(target))
        self.target = np.array(self.target, np.int32)

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Run prediction on given data")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="nli_competition.model", type=str, help="Model path")
parser.add_argument("--test_size", default=0.2, type=lambda x: int(x) if x.isdigit() else float(x),
                    help="Test set size")

def main(args):
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset("nli_dataset.train.txt")
        dev = Dataset("nli_dataset.dev.txt")

        X_train, X_test, y_train, y_test = train.data, dev.data, train.target, dev.target

        pipeline = sklearn.pipeline.Pipeline(steps=[("tf", TfidfVectorizer()), ("svm", LinearSVC())])

        params = {
            "tf__ngram_range": [(1,3)],
            "svm__C": [3, 4]
        }

        model = GridSearchCV(pipeline, params, cv=5)
        model.fit(X_train, y_train)

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