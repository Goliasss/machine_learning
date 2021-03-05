#!/usr/bin/env python3
import argparse

import numpy as np

import sklearn.datasets

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--clusters", default=5, type=int, help="Number of clusters")
parser.add_argument("--examples", default=150, type=int, help="Number of examples")
parser.add_argument("--init", default="kmeans++", type=str, help="Initialization (random/kmeans++)")
parser.add_argument("--iterations", default=5, type=int, help="Number of kmeans iterations to perfom")
parser.add_argument("--plot", default=True, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=51, type=int, help="Random seed")
# If you add more arguments, ReCodEx will keep them with your default values.

def plot(args, iteration, data, centers, clusters):
    import matplotlib.pyplot as plt

    if args.plot is not True:
        if not plt.gcf().get_axes(): plt.figure(figsize=(4*2, 5*6))
        plt.subplot(6, 2, 1 + len(plt.gcf().get_axes()))
    plt.title("KMeans Initialization" if not iteration else
              "KMeans After Initialization {}".format(iteration))
    plt.gca().set_aspect(1)
    plt.scatter(data[:, 0], data[:, 1], c=clusters)
    plt.scatter(centers[:, 0], centers[:, 1], marker="P", s=200, c="#ff0000")
    plt.scatter(centers[:, 0], centers[:, 1], marker="P", s=50, c=range(args.clusters))
    if args.plot is True: plt.show()
    else: plt.savefig(args.plot, transparent=True, bbox_inches="tight")

def main(args):
    # Create a random generator with a given seed
    generator = np.random.RandomState(args.seed)

    # Generate artificial data
    data, target = sklearn.datasets.make_blobs(
        n_samples=args.examples, centers=args.clusters, n_features=2, random_state=args.seed)

    # TODO: Initialize `centers` to be
    # - if args.init == "random", K random data points, using the indices
    #   returned by
    #     generator.choice(len(data), size=args.clusters, replace=False)
    # - if args.init == "kmeans++", generate the first cluster by
    #     generator.randint(len(data))
    #   and then iteratively sample the rest of the clusters proportionally to
    #   the square of their distances to their closest cluster using
    #     generator.choice(unused_points_indices, p=square_distances / np.sum(square_distances))
    #   Use the `np.linalg.norm` to measure the distances.
    if args.init == "random":
        indices = generator.choice(len(data), size=args.clusters, replace=False)
        centers = data[indices]
    elif args.init == "kmeans++":
        first_one = generator.randint(len(data))
        centers = [None] * args.clusters
        unused_points_indices = list(range(args.examples))
        unused_points_indices.remove(first_one)
        centers[0] = list(data[first_one])
        for k in range(1, args.clusters):
            print(k)
            square_distances = []
            for i in unused_points_indices:
                distances = [np.linalg.norm(centers[j] - data[i])**2 for j in range(k)]
                square_distances.append(np.min(distances))
            # square_distances = [np.linalg.norm(centers[k-1] - data[i])**2 for i in unused_points_indices]
            new = generator.choice(unused_points_indices, p=square_distances / np.sum(square_distances))
            centers[k] = list(data[new])
            unused_points_indices.remove(new)
        centers = np.array(centers)
        print(centers)
        pass

    if args.plot:
        plot(args, 0, data, centers, clusters=None)
    z = [None] * args.examples
    # Run `args.iterations` of the K-Means algorithm.
    for iteration in range(args.iterations):
        # TODO: Perform a single iteration of the K-Means algorithm, storing
        # zero-based cluster assignment to `clusters`.
        for i in range(args.examples):
            z[i] = np.argmin([np.linalg.norm(data[i] - centers[j]) for j in range(args.clusters)])

        for k in range(args.clusters):
            centers[k] = np.dot(np.array(z) == k, data) / np.sum(np.array(z) == k)

        clusters = z

        if args.plot:
            plot(args, 1 + iteration, data, centers, clusters)

    return clusters

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    centers = main(args)
    print("Cluster assignments:", centers, sep="\n")