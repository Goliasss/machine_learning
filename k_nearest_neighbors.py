#!/usr/bin/env python3
import argparse
import os
import sys
import urllib.request

import numpy as np
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing

class MNIST:
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
parser.add_argument("--k", default=10, type=int, help="K nearest neighbors to consider")
parser.add_argument("--p", default=2, type=int, help="Use L_p as distance metric")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=500, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
parser.add_argument("--train_size", default=5000, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
parser.add_argument("--weights", default="softmax", type=str, help="Weighting to use (uniform/inverse/softmax)")
# If you add more arguments, ReCodEx will keep them with your default values.

def main(args):
    # Load MNIST data, scale it to [0, 1] and split it to train and test
    mnist = MNIST()
    mnist.data = sklearn.preprocessing.MinMaxScaler().fit_transform(mnist.data)
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        mnist.data, mnist.target, stratify=mnist.target, train_size=args.train_size, test_size=args.test_size, random_state=args.seed)

    # TODO: Generate `test_predictions` with classes predicted for `test_data`.
    #
    # Find `args.k` nearest neighbors, choosing the ones with smallest train_data
    # indices in case of ties. Use the most frequent class (optionally weighted
    # by a given scheme described below) as prediction, choosing the one with the
    # smallest class index when there are multiple classes with the same frequency.
    #
    # Use L_p norm for a given p (1, 2, 3) to measure distances.
    #
    # The weighting can be:
    # - "uniform": all nearest neighbors have the same weight
    # - "inverse": `1/distances` is used as weights
    # - "softmax": `softmax(-distances)` is uses as weights
    #
    # If you want to plot misclassified examples, you need to also fill `test_neighbors`
    # with indices of nearest neighbors; but it is not needed for passing in ReCodEx.

    encoder = sklearn.preprocessing.OneHotEncoder(sparse = False)
    train_target = train_target.reshape(-1,1)
    train_target_encoded = encoder.fit_transform(train_target)

    def softmax(y, i):
        y = y - np.amax(y)
        # print(y)
        return np.exp(y[i]) / np.sum([np.exp(y[j]) for j in range(args.k)])

    def get_weights(distances, type):
        if type == "uniform":
            return np.ones(shape=distances.shape)
        if type == "inverse":
            return 1/distances
        if type == "softmax":
            results = np.ones(shape=distances.shape)
            for i in range(distances.shape[0]):
                results[i] = softmax(-distances, i)
            results = np.array(results)
            return results

    test_predictions = []
    distances = np.empty(shape=[test_data.shape[0], train_data.shape[0]])
    for i in range(len(test_data)):
        for j in range(len(train_data)):
            distances[i, j] = np.linalg.norm(test_data[i]-train_data[j], ord=args.p)
        nearest_indexes = np.argpartition(distances[i,:], args.k)[0:args.k]                 # ties?
        weights = get_weights(distances[i, nearest_indexes], args.weights)

        sum_of_all = np.sum(weights[j] for j in range(args.k))
        train_target_encoded[nearest_indexes]
        prediction = np.sum(weights[i]*train_target_encoded[nearest_indexes][i]/sum_of_all for i in range(args.k))
        prediction = np.argmax(prediction)
        test_predictions = np.append(test_predictions, prediction)



    # test_predictions = None

    accuracy = sklearn.metrics.accuracy_score(test_target, test_predictions)

    if args.plot:
        import matplotlib.pyplot as plt
        examples = [[] for _ in range(10)]
        for i in range(len(test_predictions)):
            if test_predictions[i] != test_target[i] and not examples[test_target[i]]:
                examples[test_target[i]] = [test_data[i], *train_data[test_neighbors[i]]]
        examples = [[img.reshape(28, 28) for img in example] for example in examples if example]
        examples = [[example[0]] + [np.zeros_like(example[0])] + example[1:] for example in examples]
        plt.imshow(np.concatenate([np.concatenate(example, axis=1) for example in examples], axis=0), cmap="gray")
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().get_yaxis().set_visible(False)
        if args.plot is True: plt.show()
        else: plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return accuracy

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    accuracy = main(args)
    print("K-nn accuracy for {} nearest neighbors, L_{} metric, {} weights: {:.2f}%".format(
        args.k, args.p, args.weights, 100 * accuracy))
