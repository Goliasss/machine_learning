#!/usr/bin/env python3
import argparse
import lzma
import pickle
import os
import urllib.request
import sys
import time

import numpy as np

import sklearn
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.neural_network
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.compose
import sklearn.svm
class Dataset:
    """MNIST Dataset.
    The train set contains 60000 images of handwritten digits. The data
    contain 28*28=784 values in range 0-255, the targets are numbers 0-9.
    """
    def __init__(self,
                 name="mnist.train.npz",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2021/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset, i.e., `data` and optionally `target`.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value)
        self.data = self.data.reshape([-1, 28*28]).astype(np.float)


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Run prediction on given data")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="mnist_competition.model", type=str, help="Model path")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x),
                    help="Test set size")
def main(args):
    if args.predict is None:
        # We are training a model.
        start_time = time.time()
        np.random.seed(args.seed)
        train = Dataset()

        data = np.append(train.data, np.repeat([[1]], train.data.shape[0], axis=0), axis=1)

        train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(data, train.target,
                                                                                                    test_size=args.test_size,
                                                                                                    random_state=args.seed)

        # TODO: Train a model on the given dataset and store it in `model`.
        # pipeline = sklearn.pipeline.Pipeline([('MLP', sklearn.neural_network.MLPClassifier())])
        #
        # parameteres = {'MLP__activation': ['tanh'], 'MLP__alpha': [0.8], 'MLP__hidden_layer_sizes': [[100, 50]], 'MLP__learning_rate': ['adaptive']}

        pipeline = sklearn.pipeline.Pipeline([('SVC', sklearn.svm.SVC(gamma='auto'))])
        parameteres = {'SVC__kernel': ['poly'], 'SVC__C': [0.5], 'SVC__degree': [2]}

        model = sklearn.model_selection.GridSearchCV(pipeline, param_grid=parameteres, cv=5)
        model.fit(train_data, train_target)
        print(model.best_params_)

        print("--- %s seconds ---" % (time.time() - start_time))
        test_predicted = model.predict(test_data)

        test_accuracy = sum(test_predicted == test_target) / len(test_target)

        print(test_accuracy)
        print(model.best_params_)
        print(train_data.shape)
        # If you trained one or more MLPs, you can use the following code
        # to compress it significantly (approximately 12 times). The snippet
        # assumes the trained MLPClassifier is in `mlp` variable.
        # mlp._optimizer = None
        # for i in range(len(mlp.coefs_)): mlp.coefs_[i] = mlp.coefs_[i].astype(np.float16)
        # for i in range(len(mlp.intercepts_)): mlp.intercepts_[i] = mlp.intercepts_[i].astype(np.float16)
        print("--- %s seconds ---" % (time.time() - start_time))
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