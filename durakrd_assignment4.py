# Assignment 4
# Ling471 Spring 2023

import pandas as pd
import string
import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# These are your own functions you wrote for Assignment 3:
from durakrd_assignment3 import computePrecisionRecall, computeAccuracy, POS_REVIEW, NEG_REVIEW

# Note: If you did not put your functions into the `evaluation.py` file
# import the functions however you need, or copy and paste them into this file
# (and you would then need  to delete the `from evaluation import...` line above)

# Constants
ROUND = 4
GOOD_REVIEW = 1
BAD_REVIEW = 0
ALPHA = 1

# This function will be reporting errors due to variables which were not assigned any value.
# Your task is to get it working! You can comment out things which aren't working at first.
def main(argv):
    # Read in the data. NB: You may get an extra Unnamed column with indices; this is OK.
    # If you like, you can get rid of it by passing a second argument to the read_csv(): index_col=[0].
    data = pd.read_csv(argv[1], index_col=[0])
    # print(data.head())  # <- Verify the format. Comment this back out once done.

    # TODO: Change as appropriate, if you stored data differently (e.g. if you put train data first).
    # You may also make use of the "type" column here instead! E.g. you could sort data by "type".
    # At any rate, make sure you are grabbing the right data! Double check with temporary print statements,
    # e.g. print(test_data.head()).
    data['label'] = data['label'].replace(['pos', 'neg'], [POS_REVIEW, NEG_REVIEW])

    test_data = data[:25000]  # Assuming the first 25,000 rows are test data.

    # Assuming the second 25,000 rows are training data. Double check!
    train_data = data[25000:50000]

    # TODO: Set the below 4 variables to contain:
    # X_train: the training data; y_train: the training data labels;
    # X_test: the test data; y_test: the test data labels.
    # Access the data frames by the appropriate column names.
    X_train = train_data["review"]
    y_train = train_data["label"]

    X_test = test_data["review"]
    y_test = test_data["label"]

    # The next three lines are performing feature extraction and word counting. 
    # They are choosing which words to count frequencies for, basically, to discard some of the noise.
    # If you are curious, you could read about TF-IDF,
    # e.g. here: https://www.geeksforgeeks.org/tf-idf-model-for-page-ranking/
    # or here: https://en.wikipedia.org/wiki/Tf%E2%80%93idf

    # TODO: Add a general brief comment on why choosing which words to count may be important.
    # Properly weighing words that are used frequently but do not signify anything meaningful by themselves is important
    # to prevent overfitting. Words such as "the" and "there" are used frequently and are inaccurate words to use to
    # judge a review. Weighing them using the tf_idf vectorizer can make the models accurate and efficient.
    # Additionally, the term frequency is used for laplace smoothing.

    tf_idf_vect = TfidfVectorizer(ngram_range=(1, 2))
    tf_idf_train = tf_idf_vect.fit_transform(X_train.values)
    tf_idf_test = tf_idf_vect.transform(X_test.values)

    # TODO COMMENT: The hyperparameter alpha is used for Laplace Smoothing.
    # Add a brief comment, trying to explain, in your own words, what smoothing is for.
    # You may want to read about Laplace smoothing here: https://towardsdatascience.com/laplace-smoothing-in-na%C3%AFve-bayes-algorithm-9c237a8bdece
    # Laplace smoothing is used for instances where a word that appears in the test dataset was not used to train the
    # model. This would result in a zero probability in Naïve Bayes. Without laplace smoothing, the word would
    # significantly alter the prediction the model makes. The term frequency is used to make an accurate assessment.

    clf = MultinomialNB(alpha=ALPHA)

    # TODO COMMENT: Add a comment explaining in your own words what the "fit()" method is doing.
    # The fit() method uses the tf_idf and laplace smoothing algorithms to create a model. The model takes in values
    # with their text weighted frequency and their corresponding gold labels. The model then "weighs" words to determine
    # the likelihood it is associated with a positive or negative review.

    clf.fit(tf_idf_train, y_train)

    # TODO COMMENT: Add a comment explaining in your own words what the "predict()" method is doing in the next two lines.
    # The predict() method takes in reviews that have have had their associated text weighting frequency and outputs
    # corresponding predictions. The methods runs reviews or "values" through the model and perform Naïve Bayes
    # probability calculations to determine whether a review is positive or negative.

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

    # Report the metrics via standard output.
    # Please DO NOT modify the format (for grading purposes).
    # You may change the variable names of course, if you used different ones above.

    print("Train accuracy:           \t{}".format(round(accuracy_train, ROUND)))
    print("Train precision positive: \t{}".format(
        round(precision_pos_train, ROUND)))
    print("Train recall positive:    \t{}".format(
        round(recall_pos_train, ROUND)))
    print("Train precision negative: \t{}".format(
        round(precision_neg_train, ROUND)))
    print("Train recall negative:    \t{}".format(
        round(recall_neg_train, ROUND)))
    print("Test accuracy:            \t{}".format(round(accuracy_test, ROUND)))
    print("Test precision positive:  \t{}".format(
        round(precision_pos_test, ROUND)))
    print("Test recall positive:     \t{}".format(
        round(recall_pos_test, ROUND)))
    print("Test precision negative:  \t{}".format(
        round(precision_neg_test, ROUND)))
    print("Test recall negative:     \t{}".format(
        round(recall_neg_test, ROUND)))


if __name__ == "__main__":
    main(sys.argv)
