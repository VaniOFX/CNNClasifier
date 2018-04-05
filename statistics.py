import csv
import re
import collections
import math
from preprocess import clean_str

sent_number = 0
word_number = 0
words_list = []
max_sent_len = 0
max_sent = []
number_posts = 0
max_post_len = 0
max_post = []


with open('train.csv') as test:
    next(test)
    r = csv.reader(test, delimiter=',')
    for l in r:
        number_posts += 1

        l[3] = clean_str(l[3])

        #max post length
        if len(l[3].split()) > max_post_len:
            max_post_len = len(l[3].split())
            max_post = l[3]

        #split the sentences in a post
        sents = re.split("[.;?!]+", l[3])

        #the number of sentences
        sent_number += len(sents)

        #the maximum sentence length
        for sent in sents:
            if len(sent.split()) > max_sent_len:
                max_sent_len = len(sent.split())
                max_sent = sent

        #get all the words
        words = re.split("\W+", l[3])
        word_number += len(words)
        words_list += words


    mean = int(word_number / sent_number)

with open('train.csv') as test:
    r = csv.reader(test, delimiter=',')
    #find the Standard Deviation
    total_dev = 0
    for l in r:
        total_dev += len(l[3].split()) - mean

    standard_deviation = int(math.sqrt(total_dev ** 2 / sent_number))

if __name__ == "__main__":
    print("The number of posts is", number_posts)
    print("The maximum post length is", max_post_len)
    print(max_post)
    print("The number of sentences is", sent_number)
    print("The maximum sentence length is", max_sent_len)
    print(max_sent)
    print("\nThe average sentence length is", mean)
    print("The most common words are\n", collections.Counter(words_list).most_common(10))
    print("The number of unique words is", len(set(words_list)))
    print("The number of words is", word_number)
    print("The sentence standart deviation is", standard_deviation)







