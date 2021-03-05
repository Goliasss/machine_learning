#!/usr/bin/env python3
import argparse
import heapq
import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import queue

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--criterion", default="gini", type=str, help="Criterion to use; either `gini` or `entropy`")
parser.add_argument("--max_depth", default=None, type=int, help="Maximum decision tree depth")
parser.add_argument("--max_leaves", default=None, type=int, help="Maximum number of leaf nodes")
parser.add_argument("--min_to_split", default=2, type=int, help="Minimum examples required to split")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=42, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.

def main(args):
    # Use the wine dataset
    data, target = sklearn.datasets.load_wine(return_X_y=True)

    # Split the data randomly to train and test using `sklearn.model_selection.train_test_split`,
    # with `test_size=args.test_size` and `random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # TODO: Manually create a decision tree on the trainining data.
    #
    # - For each node, predict the most frequent class (and the one with
    #   smallest index if there are several such classes).

    classes = len(np.unique(train_target))
    num_leaves = 0
    heap = []

    class Node:
        def __init__(self, data):
            self.left = None
            self.right = None
            self.data = data

    def make_split(data, target):
        return 0

    def gini(indices):
        p_T = []
        for k in range(classes):
            p_T.append(np.sum(train_target[indices] == k) / len(train_target[indices]))
        return len(train_target[indices]) * np.sum([p_T[k]*(1-p_T[k]) for k in range(classes)])

    def find_best_split(gini_score, indices, depth):
        gini_values = [[] for _ in range(train_data[indices].shape[1])]
        tresholds = [[] for _ in range(train_data[indices].shape[1])]
        for d in range(train_data[indices].shape[1]):
            for i in range(train_data[indices].shape[0] - 1):
                treshold = (train_data[indices][i][d] + train_data[indices][i + 1][d]) / 2
                gini_left = gini(np.where(train_data[indices][:, d] < treshold))
                gini_right = gini(np.where(train_data[indices][:, d] >= treshold))
                gini_values[d].append(gini_left + gini_right)
                tresholds[d].append(treshold)
        gini_min_groups = [np.argmin(gini_values[d]) for d in range(train_data[indices].shape[1])]
        treshold_feature = np.argmin(gini_min_groups)
        new_treshold = tresholds[treshold_feature][np.min(gini_min_groups)]

        left_indices = np.where(train_data[indices][:, treshold_feature]) < new_treshold
        right_indices = np.where(train_data[indices][:, treshold_feature]) >= new_treshold

        left_node = (gini(left_indices), left_indices, depth+1)
        right_node = (gini(right_indices), right_indices, depth+1)
        gini_difference = gini(left_indices) + gini(right_indices) - gini(indices)

        children = (gini_difference, left_node, right_node)

        if args.max_depth is None:
            heapq.heappush(heap, children)
        elif depth < args.max_depth:
            heapq.heappush(heap, children)

        return None

    def make_tree(data, target):
        num_leaves = 0
        indices = list(range(len(train_data)))
        find_best_split(gini_score=gini(indices), indices=indices, depth=1, parent)

        while num_leaves < args.max_leaves:
            _, new_left_node, new_right_node = heapq.heappop(heap)
            find_best_split(new_left_node)
            find_best_split(new_right_node)
            num_leaves += 1


    # - When splitting a node, consider the features in sequential order, then
    #   for each feature consider all possible split points ordered in ascending
    #   value, and perform the first encountered split descreasing the criterion
    #   the most. Each split point is an average of two nearest unique feature values
    #   of the instances corresponding to the given node (i.e., for four instances
    #   with values 1, 7, 3, 3 the split points are 2 and 5).
    #
    # - Allow splitting a node only if:
    #   - when `args.max_depth` is not None, its depth must be less than `args.max_depth`;
    #     depth of the root node is zero;
    #   - there are at least `args.min_to_split` corresponding instances;
    #   - the criterion value is not zero.
    #
    # - When `args.max_leaves` is None, use recursive (left descendants first, then
    #   right descendants) approach, splitting every node if the constraints are valid.
    #   Otherwise (when `args.max_leaves` is not None), always split a node where the
    #   constraints are valid and the overall criterion value (c_left + c_right - c_node)
    #   decreases the most. If there are several such nodes, choose the one
    #   which was created sooner (a left child is considered to be created
    #   before a right child).

    # TODO: Finally, measure the training and testing accuracy.
    train_accuracy = None
    test_accuracy = None

    return train_accuracy, test_accuracy

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    train_accuracy, test_accuracy = main(args)

    print("Train accuracy: {:.1f}%".format(100 * train_accuracy))
    print("Test accuracy: {:.1f}%".format(100 * test_accuracy))