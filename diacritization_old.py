#!/usr/bin/env python3
import argparse
import lzma
import pickle
import os
import urllib.request
import sys

from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import sklearn.neighbors
from itertools import product

class Dataset:
    LETTERS_NODIA = "acdeeinorstuuyz"
    LETTERS_DIA = "áčďéěíňóřšťúůýž"

    # A translation table usable with `str.translate` to rewrite characters with dia to the ones without them.
    DIA_TO_NODIA = str.maketrans(LETTERS_DIA + LETTERS_DIA.upper(), LETTERS_NODIA + LETTERS_NODIA.upper())

    def __init__(self,
                 name="fiction-train.txt",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2021/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)
            urllib.request.urlretrieve(url + name.replace(".txt", ".LICENSE"), filename=name.replace(".txt", ".LICENSE"))

        # Load the dataset and split it into `data` and `target`.
        with open(name, "r", encoding="utf-8") as dataset_file:
            self.target = dataset_file.read()
        self.data = self.target.translate(self.DIA_TO_NODIA)

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Run prediction on given data")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="diacritization.model", type=str, help="Model path")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x),
                    help="Test set size")
parser.add_argument("--weights", default="softmax", type=str, help="Weighting to use (uniform/inverse/softmax)")

def levenshtein(token1, token2):
    distances = np.zeros((len(token1) + 1, len(token2) + 1))

    for t1 in range(len(token1) + 1):
        distances[t1][0] = t1

    for t2 in range(len(token2) + 1):
        distances[0][t2] = t2

    a = 0
    b = 0
    c = 0

    for t1 in range(1, len(token1) + 1):
        for t2 in range(1, len(token2) + 1):
            if (token1[t1 - 1] == token2[t2 - 1]):
                distances[t1][t2] = distances[t1 - 1][t2 - 1]
            else:
                a = distances[t1][t2 - 1]
                b = distances[t1 - 1][t2]
                c = distances[t1 - 1][t2 - 1]

                if (a <= b and a <= c):
                    distances[t1][t2] = a + 1
                elif (b <= a and b <= c):
                    distances[t1][t2] = b + 1
                else:
                    distances[t1][t2] = c + 1

    return distances[len(token1)][len(token2)]

def hamming_distance(chaine1, chaine2):
    return sum(c1 != c2 for c1, c2 in zip(chaine1, chaine2))

def main(args):
    if args.predict is None:
        # We are training a model.
        now = datetime.now()

        np.random.seed(args.seed)
        train = Dataset()

        tokenizer = CountVectorizer().build_tokenizer()
        data_tokens = np.array(tokenizer(train.data))
        target_tokens = np.array(tokenizer(train.target))

        data_tokens = data_tokens[np.unique(target_tokens, return_index=True)[1]]
        target_tokens = np.unique(target_tokens)

        data_tokens = np.array(sorted(data_tokens, key=len))
        target_tokens = np.array(sorted(target_tokens, key=len))

        data_tokens = data_tokens.reshape(-1, 1)
        target_tokens = target_tokens.reshape(-1, 1)

        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(data_tokens,
                                                                                    target_tokens,
                                                                                    test_size=args.test_size,
                                                                                    random_state=args.seed)
        lengths = [0,0,0]
        for i in range(len(data_tokens)-1):
            if len(data_tokens[i][0]) < len(data_tokens[i+1][0]):
                lengths = np.append(lengths, i)
        lengths = np.append(lengths, len(data_tokens)-1)
        lengths = np.append(lengths, len(data_tokens))

        # test_predictions = []
        # for i in range(len(x_test)):
        #     distances = []
        #     word_len = np.minimum(len(x_test[i][0]), 20)
        #     if x_test[i][0] in data_tokens[lengths[word_len]:lengths[word_len+1]]:
        #         nearest_index = list(data_tokens[lengths[word_len]:lengths[word_len+1]]).index(x_test[i][0])
        #         test_predictions = np.append(test_predictions, target_tokens[nearest_index + lengths[word_len]][0])
        #     else:
        #         for j in range(lengths[word_len], lengths[word_len+1]):
        #             dist = hamming_distance(x_test[i][0], data_tokens[j][0])
        #             distances = np.append(distances, dist)
        #             if dist == 0:
        #                 break
        #         nearest_index = np.argmin(distances)
        #         test_predictions = np.append(test_predictions, target_tokens[nearest_index+lengths[word_len]][0])
        #     print(x_test[i][0], test_predictions[i])
        #
        # dictionary = dict(zip(list(x_test.reshape(len(x_test))), list(test_predictions)))
        #

        dictionary = dict(zip(list(x_test.reshape(len(x_test))), list(y_test.reshape(len(x_test)))))

        string = train.data[0:500]
        string_gut = train.target[0:500]
        print(string)

        temp = string.split()
        res = []
        for wrd in temp:
            # searching from lookp_dict
            res.append(dictionary.get(wrd, wrd))

        res = ' '.join(res)

        # printing result
        print(res)

        def accuracy(gold, system):
            assert isinstance(gold, str) and isinstance(system, str), "The gold and system outputs must be strings"
            gold, system = gold.split(), system.split()
            assert len(gold) == len(
                system), "The gold and system outputs must have same number of words: {} vs {}.".format(len(gold),
                                                                                                        len(system))

            words, correct = 0, 0
            for gold_token, system_token in zip(gold, system):
                words += 1
                correct += gold_token == system_token

            return correct / words
        print(accuracy(string_gut, res))

        # for i in range(len(x_test)):
        #     if str(x_test[i][0]) in string:
        #         print("yes")
        #         # string.replace(str(x_test[i][0]), str(test_predictions[i]))
        #         string = string.replace(str(x_test[i][0]), str(y_test[i][0]))
        #     else:
        #         print("no")
        # print(string)

        timestamp = datetime.timestamp(now)
        print("timestamp =", timestamp)
        # TODO: Train a model on the given dataset and store it in `model`.
        model = train

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions, as either a Python list or a NumPy array.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        train = model

        tokenizer = CountVectorizer().build_tokenizer()
        data_tokens = np.array(tokenizer(train.data))
        target_tokens = np.array(tokenizer(train.target))

        data_tokens = data_tokens[np.unique(target_tokens, return_index=True)[1]]
        target_tokens = np.unique(target_tokens)

        data_tokens = np.array(sorted(data_tokens, key=len))
        target_tokens = np.array(sorted(target_tokens, key=len))

        data_tokens = data_tokens.reshape(-1, 1)
        target_tokens = target_tokens.reshape(-1, 1)

        x_test = np.array(tokenizer(test.data))
        x_test = np.unique(x_test)
        x_test = x_test.reshape(-1, 1)

        lengths = [0, 0, 0]
        for i in range(len(data_tokens)-1):
            if len(data_tokens[i][0]) < len(data_tokens[i+1][0]):
                lengths = np.append(lengths, i)
        lengths = np.append(lengths, len(data_tokens)-1)
        lengths = np.append(lengths, len(data_tokens))

        test_predictions = []
        for i in range(len(x_test)):
            distances = []
            word_len = len(x_test[i][0])
            if x_test[i][0] in data_tokens[lengths[word_len]:lengths[word_len+1]]:
                nearest_index = list(data_tokens[lengths[word_len]:lengths[word_len+1]]).index(x_test[i][0])
                test_predictions = np.append(test_predictions, target_tokens[nearest_index + lengths[word_len]][0])
            else:
                for j in range(lengths[word_len], lengths[word_len+1]):
                    dist = hamming_distance(x_test[i][0], data_tokens[j][0])
                    distances = np.append(distances, dist)
                    if dist == 0:
                        break
                nearest_index = np.argmin(distances)
                test_predictions = np.append(test_predictions, target_tokens[nearest_index+lengths[word_len]][0])

        string = test.data

        dictionary = dict(zip(list(x_test.reshape(len(x_test))), list(test_predictions)))

        print(string)

        temp = string.split()
        res = []
        for wrd in temp:
            # searching from lookp_dict
            res.append(dictionary.get(wrd, wrd))

        res = ' '.join(res)

        # TODO: Generate `predictions` with the test set predictions. Specifically,
        # produce a diacritized `str` with exactly the same number of words as `test.data`.
        predictions = res

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)