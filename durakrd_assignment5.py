import pandas as pd
import string
import os
import sys

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer

import numpy as np
import matplotlib.pyplot as plt

ROUND = 4
GOOD_REVIEW = 1
BAD_REVIEW = 0
ALPHA = 1

# TODO: Your custom imports here; or copy the functions to here manually.
from durakrd_assignment3 import computeAccuracy, computePrecisionRecall, POS_REVIEW, NEG_REVIEW

# TODO: You may need to modify assignment 4 if you just had a main() there.
# my_naive_bayes() should take a column as input and return as output 10 floats (numbers)
# representing the metrics.

def my_naive_bayes(subset):
    test_data = subset[:25000]  # Assuming the first 25,000 rows are test data.

    # Assuming the second 25,000 rows are training data. Double check!
    train_data = subset[25000:50000]

    # TODO: Set the below 4 variables to contain:
    # X_train: the training data; y_train: the training data labels;
    # X_test: the test data; y_test: the test data labels.
    # Access the data frames by the appropriate column names.
    X_train = train_data[train_data.columns[0]]
    y_train = train_data["label"]

    X_test = test_data[test_data.columns[0]]
    y_test = test_data["label"]

    tf_idf_vect = TfidfVectorizer(ngram_range=(1, 2))
    tf_idf_train = tf_idf_vect.fit_transform(X_train.values)
    tf_idf_test = tf_idf_vect.transform(X_test.values)

    clf = MultinomialNB(alpha=ALPHA)

    clf.fit(tf_idf_train, y_train)

    y_pred_train = clf.predict(tf_idf_train)
    y_pred_test = clf.predict(tf_idf_test)

    # TODO: Compute accuracy, precision, and recall, for both train and test data.
    # Import and call your methods from evaluation.py (or wherever) which you wrote for HW3.
    # Note: If your methods there accept lists, you will probably need to cast your pandas label objects to simple python lists:
    # e.g. list(y_train) -- when passing them to your accuracy and precision and recall functions.

    test_tuple_list = list(zip(y_pred_test, y_test))
    train_tuple_list = list(zip(y_pred_train, y_train))

    accuracy_test = computeAccuracy(test_tuple_list)
    accuracy_train = computeAccuracy(train_tuple_list)
    precision_pos_test, recall_pos_test = computePrecisionRecall(test_tuple_list, POS_REVIEW)
    precision_neg_test, recall_neg_test = computePrecisionRecall(test_tuple_list, NEG_REVIEW)
    precision_pos_train, recall_pos_train = computePrecisionRecall(train_tuple_list, POS_REVIEW)
    precision_neg_train, recall_neg_train = computePrecisionRecall(train_tuple_list, NEG_REVIEW)

    return ({
        'TRAIN': {'accuracy': accuracy_train, 'POS': {'precision': precision_pos_train, 'recall': recall_pos_train},
                  'NEG': {'precision': precision_neg_train, 'recall': recall_neg_train}},
        'TEST': {'accuracy': accuracy_test, 'POS': {'precision': precision_pos_test, 'recall': recall_pos_test},
                 'NEG': {'precision': precision_neg_test, 'recall': recall_neg_test}}
    })


def main(argv):
    data = pd.read_csv(argv[1], index_col=[0])
    data['label'] = data['label'].replace(['pos', 'neg'], [POS_REVIEW, NEG_REVIEW])
    # print(data.head())  # <- Verify the format. Comment this back out once done.

    # Part II:
    # Run all models and store the results in variables (dicts).
    # TODO: Make sure you imported your own naive bayes function and it works properly with a named column input!
    # TODO: See also the next todo which gives an example of a convenient output for my_naive_bayes()
    # which you can then easily use to collect different scores.
    # For example (and as illustrated below), the models (nb_original, nb_cleaned, etc.) can be not just lists of scores
    # but dicts where each score will be stored by key, like [TEST][POS][RECALL], etc.
    # But you can also just use lists, except then you must not make a mistake, which score you are accessing,
    # when you plot graphs.
    nb_original = my_naive_bayes(data[['review', 'label']])
    nb_cleaned = my_naive_bayes(data[['cleaned_review', 'label']])
    nb_lowercase = my_naive_bayes(data[['lowercased', 'label']])
    nb_no_stop = my_naive_bayes(data[['no stopwords', 'label']])
    nb_lemmatized = my_naive_bayes(data[['lemmatized', 'label']])

    # Collect accuracies and other scores across models.
    # TODO: Harmonize this with your own naive_bayes() function!
    # The below assumes that naive_bayes() returns a fairly complex dict of scores.
    # (NB: The dicts there contain other dicts!)
    # The return statement for that function looks like this:
    # return({'TRAIN': {'accuracy': accuracy_train, 'POS': {'precision': precision_pos_train, 'recall': recall_pos_train}, 'NEG': {'precision': precision_neg_train, 'recall': recall_neg_train}}, 'TEST': {'accuracy': accuracy_test, 'POS': {'precision': precision_pos_test, 'recall': recall_pos_test}, 'NEG': {'precision': precision_neg_test, 'recall': recall_neg_test}}})
    # This of course assumes that variables like "accuracy_train", etc., were assigned the right values already.
    # You don't have to do it this way; we are giving it to you just as an example.
    train_accuracies = []
    test_accuracies = []
    # (pos, neg) tuple format for all listed tuples
    train_precisions = []
    test_precisions = []
    train_recalls = []
    test_recalls = []
    # TODO: Initialize other score lists similarly. The precision and recalls, for negative and positive, train and test.
    for model in [nb_original, nb_cleaned, nb_lowercase, nb_no_stop, nb_lemmatized]:
        # TODO: See comment above about where this "model" dict comes from.
        # If you are doing something different, e.g. just a list of scores,
        # that's fine, change the below as appropriate,
        # just make sure you don't confuse where which score is.
        train_accuracies.append(model['TRAIN']['accuracy'])
        test_accuracies.append(model['TEST']['accuracy'])
        train_precisions.append((model['TRAIN']['POS']['precision'], model['TRAIN']['NEG']['precision']))
        test_precisions.append((model['TEST']['POS']['precision'], model['TEST']['NEG']['precision']))
        train_recalls.append((model['TRAIN']['POS']['recall'], model['TRAIN']['NEG']['recall']))
        test_recalls.append((model['TEST']['POS']['recall'], model['TEST']['NEG']['recall']))
        # TODO: Collect other scores similarly. The precision and recalls, for negative and positive, train and test.

    # TODO: Create the plot(s) that you want for the report using matplotlib (plt).
    # Use the below to save pictures as files:
    ordered_list = ["nb_original", "nb_cleaned", "nb_lowercase", "nb_no_stop", "nb_lemmatized"]
    # Train Accuracies Plot
    plt.bar(ordered_list, train_accuracies)
    plt.xlabel("Models")
    plt.ylabel("Percent")
    plt.title("Train Accuracies")
    plt.savefig('train_accuracies.png')
    plt.clf()

    # Test Accuracies Plot
    plt.bar(ordered_list, test_accuracies)
    plt.xlabel("Models")
    plt.ylabel("Percent")
    plt.title("Test Accuracies")
    plt.ylim([0, 1])
    plt.savefig('test_accuracies.png')
    plt.clf()


    # Train Precision Plot
    width = 0.3
    plt.bar(np.arange(5), [x[0] for x in train_precisions], width=width, label="Positive")
    plt.bar(np.arange(5) + width, [x[1] for x in train_precisions], width=width, label="Negative")
    plt.xlabel("Models")
    plt.ylabel("Percent")
    plt.title("Train Precisions")
    plt.xticks([r + width for r in range(5)], ordered_list)
    plt.legend()
    plt.ylim([0, 1])
    plt.savefig('train_precisions.png')
    plt.clf()

    #Test Precision Plot
    plt.bar(np.arange(5), [x[0] for x in test_precisions], width=width, label="Positive")
    plt.bar(np.arange(5) + width, [x[1] for x in test_precisions], width=width, label="Negative")
    plt.xlabel("Models")
    plt.ylabel("Percent")
    plt.title("Test Precisions")
    plt.xticks([r + width for r in range(5)], ordered_list)
    plt.legend()
    plt.ylim([0, 1])
    plt.savefig('test_precisions.png')
    plt.clf()

    #Train Recall Plot
    plt.bar(np.arange(5), [x[0] for x in train_recalls], width=width, label="Positive")
    plt.bar(np.arange(5) + width, [x[1] for x in train_recalls], width=width, label="Negative")
    plt.xlabel("Models")
    plt.ylabel("Percent")
    plt.title("Train Recalls")
    plt.xticks([r + width for r in range(5)], ordered_list)
    plt.legend()
    plt.ylim([0, 1])
    plt.savefig('train_recalls.png')
    plt.clf()

    #Test Recall Plot
    plt.bar(np.arange(5), [x[0] for x in test_recalls], width=width, label="Positive")
    plt.bar(np.arange(5) + width, [x[1] for x in test_recalls], width=width, label="Negative")
    plt.xlabel("Models")
    plt.ylabel("Percent")
    plt.title("Test Recalls")
    plt.xticks([r + width for r in range(5)], ordered_list)
    plt.legend()
    plt.ylim([0, 1])
    plt.savefig('test_recalls.png')
    plt.clf()


if __name__ == "__main__":
    main(sys.argv)
