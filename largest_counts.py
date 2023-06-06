import sys
import pandas as pd

# From Assignment 2, copied manually here just to remind you
# that you can copy stuff manually if importing isn't working out.
# You can just use this or you can replace it with your function.

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

    # TODO: Write a for loop here, doing what is described above.
    for word in tokens:
        if word not in token_counts:
            token_counts[word] = 0
        token_counts[word] += 1

    # sorts dictionaries for largest counts
    sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
    return dict(sorted_tokens)



def largest_counts(data):  # TODO: Finish implementing this function

    # TODO: Cut up the rows in the dataset according to how you stored things.
    # The below assumes test data is stored first and negative is stored before positive.
    # If you did the same, no change is required.
    train_neg_data = data[(data["label"] == "neg") & (data["type"] == "train")]
    train_pos_data = data[(data["label"] == "pos") & (data["type"] == "train")]
    # TODO: SORT the count dicts which countTokens() returns
    # by value (count) in reverse (descending) order.
    # It is your task to Google and learn how to do this, but we will help of course,
    # if you come to use with questions. This can be daunting at first, but give it time.
    # Spend some (reasonable) time across a few days if necessary, and you will do it!

    # As is, the counts returned by the counter AREN'T sorted!
    # So you won't be able to easily retrieve the most frequent words.

    # NB: str.cat() turns whole column into one text
    counts_pos_original = countTokens(
        train_pos_data["review"].str.cat())
    counts_pos_cleaned = countTokens(
        train_pos_data["cleaned_review"].str.cat())
    counts_pos_lowercased = countTokens(
        train_pos_data["lowercased"].str.cat())
    counts_pos_no_stop = countTokens(
        train_pos_data["no stopwords"].str.cat())
    counts_pos_lemmatized = countTokens(
        train_pos_data["lemmatized"].str.cat())

    counts_neg_original = countTokens(
        train_neg_data["review"].str.cat())
    counts_neg_cleaned = countTokens(
        train_neg_data["cleaned_review"].str.cat())
    counts_neg_lowercased = countTokens(
        train_neg_data["lowercased"].str.cat())
    counts_neg_no_stop = countTokens(
        train_neg_data["no stopwords"].str.cat())
    counts_neg_lemmatized = countTokens(
        train_neg_data["lemmatized"].str.cat())

    # Once the dicts are sorted, output the first 20 rows for each.
    # This is already done below, but changes may be needed depending on what you did to sort the dicts.
    # The [:19] "slicing" syntax expects a list. If you sorting call return a list (which is likely, as being sorted
    # is conceptualy a properly of LISTS,  NOT dicts),
    # you may want to remove the additional list(dict_name.items()) conversion.
    with open('counts.txt', 'w') as f:
        f.write('Original POS reviews:\n')
        for k, v in list(counts_pos_original.items())[:20]:
            f.write('{:10s} {:.0f}\n'.format(k, v))
        f.write('Cleaned POS reviews:\n')
        for k, v in list(counts_pos_cleaned.items())[:20]:
            f.write('{:10s} {:.0f}\n'.format(k, v))
        f.write('Lowercased POS reviews:\n')
        for k, v in list(counts_pos_lowercased.items())[:20]:
            f.write('{:10s} {:.0f}\n'.format(k, v))
        f.write('No stopwords POS reviews:\n')
        for k, v in list(counts_pos_no_stop.items())[:20]:
            f.write('{:10s} {:.0f}\n'.format(k, v))
        f.write('Lemmatized POS reviews:\n')
        for k, v in list(counts_pos_lemmatized.items())[:20]:
            f.write('{:10s} {:.0f}\n'.format(k, v))

        f.write('Original NEG reviews:\n')
        for k, v in list(counts_neg_original.items())[:20]:
            f.write('{:10s} {:.0f}\n'.format(k, v))
        f.write('Cleaned NEG reviews:\n')
        for k, v in list(counts_neg_cleaned.items())[:20]:
            f.write('{:10s} {:.0f}\n'.format(k, v))
        f.write('Lowercased NEG reviews:\n')
        for k, v in list(counts_neg_lowercased.items())[:20]:
            f.write('{:10s} {:.0f}\n'.format(k, v))
        f.write('No stopwords NEG reviews:\n')
        for k, v in list(counts_neg_no_stop.items())[:20]:
            f.write('{:10s} {:.0f}\n'.format(k, v))
        f.write('Lemmatized NEG reviews:\n')
        for k, v in list(counts_neg_lemmatized.items())[:20]:
            f.write('{:10s} {:.0f}\n'.format(k, v))
        # TODO: Do the same for all the remaining training dicts, per Assignment spec.

    # TODO: Copy the output of the above print statements
    #  into your document/report, or otherwise create a table/visualization for these counts.
    # Manually is fine, or you may explore bar charts in pandas! Be creative :).


def main(argv):
    data = pd.read_csv(argv[1], index_col=[0])
    # print(data.head())  # <- Verify the format. Comment this back out once done.

    largest_counts(data)


if __name__ == "__main__":
    main(sys.argv)
