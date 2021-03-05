#!/usr/bin/env python3
import argparse

import numpy as np
from scipy.stats import norm
from sklearn.metrics import accuracy_score
import sklearn.datasets
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--alpha", default=10, type=float, help="Smoothing parameter for Bernoulli and Multinomial NB")
parser.add_argument("--naive_bayes_type", default="gaussian", type=str, help="NB type to use")
parser.add_argument("--classes", default=10, type=int, help="Number of classes")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=41, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.

def main(args):
    # Use the digits dataset.
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)

    # Split the dataset into a train set and a test set.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # TODO: Train a naive Bayes classifier on the train data.
    #
    # The `args.naive_bayes_type` can be one of:
    # - "gaussian": implement Gaussian NB training, by estimating mean and
    #   variance of the input features. For variance estimation use
    #     1/N * \sum_x (x - mean)^2
    #   and additionally increase all estimated variances by `args.alpha`.
    #
    #   During prediction, you can compute probability density function of a Gaussian
    #   distribution using `scipy.stats.norm`, which offers `pdf` and `logpdf`
    #   methods, among others.
    #
    # - "multinomial": Implement multinomial NB with smoothing factor `args.alpha`.
    #
    # - "bernoulli": Implement Bernoulli NB with smoothing factor `args.alpha`.
    #   Do not forget that Bernoulli NB works with binary data, so consider
    #   all non-zero features as ones during both estimation and prediction.

    p_C = np.zeros(args.classes)
    for k in range(args.classes):
        p_C[k] = len(train_data[train_target == k]) / len(train_data)

    if args.naive_bayes_type == "gaussian":
        mean = np.zeros(shape=(train_data.shape[1], args.classes))
        var = np.zeros(shape=(train_data.shape[1], args.classes))
        for k in range(args.classes):
            for i in range(train_data.shape[1]):
                mean[i, k] = np.mean(train_data[train_target == k][:, i])
                var[i, k] = np.var(train_data[train_target == k][:, i], ddof=0) + args.alpha

        vgauss = np.vectorize(norm.logpdf)

        def func(x):
            return norm.logpdf(x[0],x[1],x[2])

        test_prediction = []
        for j in range(len(test_data)):
            probabilities = []
            for k in range(args.classes):
                probability = np.log(p_C[k])
                sum = norm.logpdf( test_data[j,:], mean[:,k], np.sqrt(var[:,k]))
                # print(mean[:,k], np.sqrt(var[:,k]), test_data[j,:])
                probability += np.sum(sum)
                probabilities.append(probability)

            test_prediction = np.append(test_prediction, np.argmax(probabilities))

        print(test_prediction)

    elif args.naive_bayes_type == "multinomial":
        p = np.zeros(shape=(train_data.shape[1], args.classes))
        n = np.zeros(shape=(train_data.shape[1], args.classes))
        for k in range(args.classes):
            for i in range(train_data.shape[1]):
                n[i, k] = np.sum([train_data[train_target == k][j][i] for j in range(len(train_data[train_target == k]))])
        for k in range(args.classes):
            for i in range(train_data.shape[1]):
                p[i, k] = (n[i, k] + args.alpha) / (np.sum([n[j, k] for j in range(train_data.shape[1])]) + args.alpha * train_data.shape[1])

        test_prediction = []
        for j in range(len(test_data)):
            probabilities = []
            for k in range(args.classes):
                probability = np.log(p_C[k]) + np.sum([test_data[j][i] * np.log(p[i, k]) for i in range(test_data.shape[1])])
                probabilities.append(probability)
            test_prediction.append(np.argmax(probabilities))
    elif args.naive_bayes_type == "bernoulli":
        p = np.zeros(shape=(train_data.shape[1], args.classes))
        for k in range(args.classes):
            for i in range(train_data.shape[1]):
                p[i, k] = (np.sum([train_data[train_target == k][:, i] > 0]) + args.alpha) / (len(train_data[train_target == k]) + 2*args.alpha)

        test_prediction = []
        for j in range(len(test_data)):
            probabilities = []
            for k in range(args.classes):
                probability = np.log(p_C[k]) + np.sum([(test_data[j][i] > 0) * np.log((p[i, k])/(1 - p[i, k])) + np.log(1 - p[i, k]) for i in range(test_data.shape[1])])
                probabilities.append(probability)
            test_prediction.append(np.argmax(probabilities))

    # TODO: Predict the test data classes and compute test accuracy.
    test_accuracy = np.sum(test_prediction == test_target) / len(test_target)

    return test_accuracy

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    test_accuracy = main(args)

    print("Test accuracy {:.2f}%".format(100 * test_accuracy))