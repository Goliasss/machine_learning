#!/usr/bin/env python3
import argparse
import os
import sys
import urllib.request

import numpy as np
import sklearn.base
import sklearn.linear_model
import sklearn.model_selection
import sklearn.pipeline

class MNIST:
    """MNIST Dataset.
    The train set contains 60000 images of handwritten digits. The data
    contain 28*28=784 values in range 0-255, the targets are numbers 0-9.
    """
    def __init__(self,
                 name="mnist.train.npz",
                 data_size=None,
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2021/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset, i.e., `data` and optionally `target`.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value[:data_size])
        self.data = self.data.reshape([-1, 28*28]).astype(np.float)

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--max_iter", default=100, type=int, help="Maximum iterations for LR")
parser.add_argument("--pca", default=9, type=int, help="PCA dimensionality")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.

class PCATransformer(sklearn.base.TransformerMixin):
    def __init__(self, n_components, seed):
        self._n_components = n_components
        self._seed = seed

    def fit(self, X, y=None):
        generator = np.random.RandomState(self._seed)

        # TODO: Compute the `args._n_components` principal components
        if self._n_components <= 10:
            # TODO: Use the power iteration algorithm for <= 10 dimensions.
            mu = np.mean(X, axis=0)
            S = np.matmul(np.transpose(X-mu), (X-mu)) / X.shape[0]
            v = [None]*self._n_components
            lambd = [None]*self._n_components
            for i in range(self._n_components):
                v[i] = generator.uniform(-1, 1, size=X.shape[1])
                for k in range(10):
                    v[i] = np.matmul(S, v[i])
                    lambd[i] = np.linalg.norm(v[i])
                    v[i] = v[i] / lambd[i]
                S = S - lambd[i] * np.outer(v[i],v[i])

            self._V = np.column_stack(v)
            print(self._V)
            print(self._V.shape)
            pass

        else:
            # TODO: Use the SVD decomposition computed with `np.linalg.svd`
            # to find the principal components.
            U, D, V = np.linalg.svd(X-np.mean(X, axis=0))
            self._V = np.transpose(V[0:self._n_components, :])
            pass

        # We round the principal components to avoid rounding errors during
        # ReCodEx evaluation.
        self._V = np.around(self._V, decimals=4)

        return self

    def transform(self, X):
        # TODO: Transform the given `X` using the precomputed `self._V`.
        return np.matmul(X, self._V)

def main(args):
    # Use the MNIST dataset.
    dataset = MNIST(data_size=5000)

    # Split the dataset into a train set and a test set.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        dataset.data, dataset.target, test_size=args.test_size, random_state=args.seed)

    pca = [("PCA", PCATransformer(args.pca, args.seed))] if args.pca else []

    pipeline = sklearn.pipeline.Pipeline(
        [("scaling", sklearn.preprocessing.MinMaxScaler())] +
        pca +
        [("classifier", sklearn.linear_model.LogisticRegression(solver="saga", max_iter=args.max_iter, random_state=args.seed))]
    )
    pipeline.fit(train_data, train_target)

    test_accuracy = pipeline.score(test_data, test_target)
    return test_accuracy

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    accuracy = main(args)
    print("Test set accuracy: {:.2f}%".format(100 * accuracy))