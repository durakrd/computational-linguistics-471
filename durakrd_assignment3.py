# This is a skeleton for your assignment 3 script.
# It contains the program structure, function names,
# the main function

# Import modules
import sys
import re
import string
from pathlib import Path

"""
The function below should be called on a file name.It should open the file,
read its contents, and store it in a variable. Then it should remove punctuation
marks and return the "cleaned" text.
"""
def cleanFileContents(f):
    # The two lines below open the file and read all the text from it
    # storing it into a variable called "text".
    # You do not need to modify the below two lines; they are already working as needed.
    with open(f, 'r', encoding='utf-8') as f:
        text = f.read()

    # The line below will clean the text of punctuation marks.
    # Ask if you are curious about how it works! But you can just use it as is.
    # Observe the effect of the function by inspecting the debugger pane while stepping over.
    clean_text = text.translate(str.maketrans('', '', string.punctuation))

    # Now, we will want to replace all tabs with spaces, and also all occurrences of more than one
    # space in a row with a single space. Review the regular expression slides/readings, and
    # write a statement below which replaces all occurrences of one or more whitespace-group characters
    # (that will include tabs) with a single space. You want the re.sub function.
    # The shortcut for all whitespace characters is \s. The regex operator for "one or more" is +.
    # Read re.sub()'s documentation to understand which argument goes where in the parentheses.

    # TODO: Your call to the re.sub function of the regular expression module here.
    # As is, the value of clean_text does not change.
    clean_text = re.sub(r"\s\s+", " ", clean_text)

    # Do not forget to return the result!
    return clean_text


'''
The below function takes a string as input, breaks it down into word tokens by space, and stores, in a dictionary table,
how many times each word occurred in the text. It returns the dictionary table.
'''
def countTokens(text):
    # Initializing an empty dictionary. Do not modify the line below, it is already doing what is needed.
    token_counts = {}

    # Use the split() function, defined for strings, to split the text by space.
    # Store the result in a variable, e.g. called "tokens".
    # See what the split() function returns and stores in your variable
    # as you step through the execution in the debugger.

    # TODO: Write a statement below calling split() on your text and storing the
    # result in a new variable.
    tokens = text.split()
    # Now, we need to iterate over each word in the list of tokens
    # (write a for loop over the list that split() returned).
    # Inside the loop, that is, for each word, we will perform some conditional logic:
    #   If the word is not yet stored in the dictionary
    #   we called "token_counts" as a key, we will store it there now,
    #   and we will initialize the key's value to 0.
    # Outside that if statement: now that we are sure
    # the word is stored as a key, we will increment the count by 1.
    for word in tokens:
        token_counts[word] = token_counts.get(word, 0) + 1
    # Do not forget to return the result!
    return token_counts


# This silly "prediction funtion" will do the following "rudimentary data science":
# If a review contains more of the word "good" than of the word "bad", 
# the function predicts "positive" (by returning a string "POSITIVE").
# If it contains more of the word "bad" than of the word "good",
# the function predicts "negative". 
# If the count is equal (note that this includes zero count),
# the function cannot make a prediction and returns a string "NONE".


# Constants. Constants are important to avoid typo-related bugs, among other reasons.
# Use these constants as return values for the below function.

POS_REVIEW = "POSITIVE"
NEG_REVIEW = "NEGATIVE"
NONE = "NONE"
POS = 'good'
NEG = 'bad'


def predictSimplistic(filename):
    clean_text = cleanFileContents(filename)
    tokens_with_counts = countTokens(clean_text)

    # This line retrieves the count for "good". If the word "good" is not found in "counts", it returns 0.
    pos_count = tokens_with_counts.get(POS, 0)
    # TODO: Write a similar statement below to retrieve the count of "bad".
    neg_count = tokens_with_counts.get(NEG, 0)

    # TODO: Write an if-elif-else block here, following the logic described in the function description.
    # Do not forget to return the prediction! You will be returning one of the constants declared above.
    # You may choose to store a prediction in a variable and then write the return statement outside
    # of the if-else block, or you can have three return statements within the if-else block.
    if pos_count > neg_count:
        return POS_REVIEW
    elif pos_count < neg_count:
        return NEG_REVIEW
    else:
        return NONE


'''
Recommended functions to implement.
You can do things differently if you like, so long as you get correct results.

The function takes two lists (arrays), one of system predictions and one of gold labels.
The assumption is that the lists are of equal length and the order of elements in both lists
corresponds to the order of data points in the dataset. It is the responsibility of 
the function caller to ensure that is the case; otherwise the function won't return
anything meaningful.

The function computes the accuracy by comparing the predicted labels to gold labels.
Accuracy = correct predictions / total predictions
Consider also recording the indices of the data points for which a wrong prediction was made.
The function can then return a tuple: (accuracy, mistakes) where accuracy is a float
and mistakes is a list of integers.
'''
def computeAccuracy(predictions):
    # The assert statement will notify you if the condition does not hold.
    total_pred = len(predictions)

    correct_pred = 0
    for pred, gold_label in predictions:
        if pred == gold_label:
            correct_pred += 1

    accuracy = correct_pred / total_pred
    print(round(accuracy, 4))

    return NONE


'''
Recommended function spec to implement. 
You can do things differently if you like (including changing what is passed in), 
so long as you get correct results.

As suggested, the function takes three arguments. 
The first two are arrays, one of predicted labels and one of actual (gold) labels.
The third argument is a string indicating which class the precision is computed for.
This is the confusing part! You can compute precision and recall wrt the positive reviews 
or wrt the negative reviews! What is considered a "true positive" depends on what the relevant class is!

The function then computes precision as per definition: true positives / (true positives + false positives)
And it computes recall as per definition: true positives / (true positives + false negatives)

It returns a tuple of floats
: (presision, recall)
'''
def computePrecisionRecall(predictions, relevant_class):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    for pred, label in predictions:
        if pred == label and pred == relevant_class:
            true_positives += 1
        elif pred == relevant_class and pred != label:
            false_positives += 1
        elif pred != label and label == relevant_class:
            false_negatives += 1

    precision = true_positives / (true_positives + false_positives)
    print(round(precision, 4))
    recall = true_positives / (true_positives + false_negatives)
    print(round(recall, 4))
    return None


# The main function is the entry point of the program.
# When debugging, if you want to start from the very beginning,
# start here. NB: Put the breakpoint not on the "def" line but below it.
# Do not modify this function; we already wrote it for you.
# You need to modify the functions which it calls, not the main() itself.


def main(argv):
    # The directory that you will read should be passed as the argument to the program.
    # From python's point of view, it is the element number 1 in the array called argv.
    # argv is a special variable name. We don't define it ourselves; python knows about it.
    # Place the first breakpoint here, when starting.

    # pos_pred and neg_pred are lists with tuples for each review (prediction, gold_label)
    pos_path = Path(argv[1])
    pos_list = list(pos_path.glob("*.txt"))
    pos_pred = [(predictSimplistic(filename), POS_REVIEW) for filename in pos_list]

    neg_path = Path(argv[2])
    neg_list = list(neg_path.glob("*.txt"))
    neg_pred = [(predictSimplistic(filename), NEG_REVIEW) for filename in neg_list]

    full_pred = pos_pred + neg_pred

    computeAccuracy(full_pred)
    computePrecisionRecall(full_pred, POS_REVIEW)
    computePrecisionRecall(full_pred, NEG_REVIEW)


# The code below is needed so that this file can be used as a module,
# which we will (more or less) be doing in future assignments.
if __name__ == "__main__":
    main(sys.argv)
