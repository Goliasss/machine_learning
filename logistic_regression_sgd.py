#!/usr/bin/env python3
import argparse
import sys

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
parser.add_argument("--data_size", default=100, type=int, help="Data size")
parser.add_argument("--iterations", default=9, type=int, help="Number of iterations over the data")
parser.add_argument("--learning_rate", default=0.5, type=float, help="Learning rate")
parser.add_argument("--plot", default=True, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.

def classify(data):
    for i in range(len(data)):
        if data[i] >= 0.5:
            data[i] = 1
        else:
            data[i] = 0
    return data

def sigma(x):
    return 1/(1 + np.exp(-x))

def get_loss(data, target, weights):
    loss = 0
    for i in range(len(data)):
        if target[i] == 1:
            loss = loss + np.log(sigma(np.dot(data[i], weights)))
        else:
            loss = loss + np.log(1 - sigma(np.dot(data[i], weights)))
    return -loss/len(data)

def main(args):
    # Create a random generator with a given seed
    generator = np.random.RandomState(args.seed)

    # Generate an artifical regression dataset
    data, target = sklearn.datasets.make_classification(
        n_samples=args.data_size, n_features=2, n_informative=2, n_redundant=0, random_state=args.seed)

    # TODO: Append a constant feature with value 1 to the end of every input data

    data = np.append(data, np.repeat([[1]], data.shape[0], axis=0), axis=1)

    # TODO: Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.

    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(data, target,
                                                                                test_size=args.test_size,
                                                                                random_state=args.seed)

    # Generate initial linear regression weights
    weights = generator.uniform(size=train_data.shape[1])

    for iteration in range(args.iterations):
        permutation = generator.permutation(train_data.shape[0])

        # TODO: Process the data in the order of `permutation`.
        # For every `args.batch_size`, average their gradient, and update the weights.
        # You can assume that `args.batch_size` exactly divides `train_data.shape[0]`.

        for k in range(int(train_data.shape[0]/args.batch_size)):
            gradient = 0
            for i in permutation[k*args.batch_size:(k+1)*args.batch_size]:
                if train_target[i] == 1:
                    gradient = gradient + (1 - sigma(np.dot(train_data[i], weights)))*train_data[i]
                else:
                    gradient = gradient - sigma(np.dot(train_data[i], weights))*train_data[i]
            gradient = -gradient / args.batch_size
            weights = weights - args.learning_rate * gradient

        # TODO: After the SGD iteration, measure the average loss and accuracy for both the
        # train test and the test set. The loss is the average MLE loss (i.e., the
        # negative log likelihood, or crossentropy loss, or KL loss) per example.

        train_predicted = [sigma(np.dot(train_data[i], weights)) for i in range(train_data.shape[0])]
        test_predicted = [sigma(np.dot(test_data[i], weights)) for i in range(test_data.shape[0])]

        train_predicted = classify(train_predicted)
        test_predicted = classify(test_predicted)

        train_accuracy = sum(train_predicted == train_target) / len(train_target)
        test_accuracy = sum(test_predicted == test_target) / len(test_target)

        train_loss = get_loss(train_data, train_target, weights)
        test_loss = get_loss(test_data, test_target, weights)

        print("After iteration {}: train loss {:.4f} acc {:.1f}%, test loss {:.4f} acc {:.1f}%".format(
            iteration + 1, train_loss, 100 * train_accuracy, test_loss, 100 * test_accuracy))

        if args.plot:
            import matplotlib.pyplot as plt
            if args.plot is not True:
                if not iteration: plt.figure(figsize=(6.4*3, 4.8*(args.iterations+2)//3))
                plt.subplot(3, (args.iterations+2)//3, 1 + iteration)
            xs = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), 50)
            ys = np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), 50)
            predictions = [[1 / (1 + np.exp(-([x, y, 1] @ weights))) for x in xs] for y in ys]
            plt.contourf(xs, ys, predictions, levels=21, cmap=plt.cm.RdBu, alpha=0.7)
            plt.contour(xs, ys, predictions, levels=[0.25, 0.5, 0.75], colors="k")
            plt.scatter(train_data[:, 0], train_data[:, 1], c=train_target, marker="P", label="train", cmap=plt.cm.RdBu)
            plt.scatter(test_data[:, 0], test_data[:, 1], c=test_target, label="test", cmap=plt.cm.RdBu)
            plt.legend(loc="upper right")
            if args.plot is True: plt.show()
            else: plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return weights

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    weights = main(args)
    print("Learned weights", *("{:.2f}".format(weight) for weight in weights))
