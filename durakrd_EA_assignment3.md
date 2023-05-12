# Part 4: Error Analysis
1. Take note of 5 positive and 5 negative reviews which were classified
incorrectly. (You can collect the filenames for which the prediction is
incorrect, in a list, and then inspect it in the debugger.) Inspect the
contents of these 10 files. What do you notice? What kind of phenomena
(syntactic, semantic) can you observe which led to your system mistaking a
good review for a bad review, and vice versa? 

#### **Incorrectly Labelled Reviews**

|       File        | Gold Label | Prediction |
|:-----------------:|:----------:|:----------:|
|      0_9.txt      |  POSITIVE  |    NONE    |
|      2_9.txt      |  POSITIVE  |    NONE    |
|     3_10.txt      |  POSITIVE  |    NONE    |
|      4_8.txt      |  POSITIVE  |    NONE    |
|     5_10.txt      |  POSITIVE  |    NONE    |
|     143_2.txt     |  NEGATIVE  |    NONE    |
|     147_4.txt     |  NEGATIVE  |  POSITIVE  |
|     149_1.txt     |  NEGATIVE  |    NONE    |
|     151_1.txt     |  NEGATIVE  |  POSITIVE  |
|     152_4.txt     |  NEGATIVE  |    NONE    |

**In positive reviews, the word "good" does not appear as often in them.
Rather, descriptive adjectives with a positive meaning are used to highlight
aspects the reviewers like. In positive reviews, the words "good" and "bad"
are both not used often therefore causing them to be flagged as "NONE" by
the predictSimplistic function. Negative movies reviews tend to have the word
"good" used before a negative word. For instance, "There is nothing good in the movie".
Additionally, the word "good" is often used in negative reviews to provide kind remarks to
balance their harsh critiques.**

2. Why do you think the precision is so much better than the recall?

**Precision is better than recall because there are fewer reported false positives
than false negatives. This means that the predictSimplistic function has a higher amount of
predictive labels match the gold labels. The function is also less likely to label actual positive/negative
reviews correctly. Precision is so much better than recall because reviews are likely to be predicted accurately
if they have the words "good" and "bad". However, recall is worse because most positive/negative reviews do not
have those words in them.**

3. Why do you think the precision for negative reviews may be so much better
than for positive reviews?

**Precision might be better for negative reviews when compared to  positive ones because the words "good" and
"bad" are not distributed evenly. Negative reviews are more likely to use the word "bad" whereas "good" is not
used as often in positive reviews. This might be because positive reviews may use more descriptive words to highlight
the positive aspects of a movie.**