import csv
import re
import collections
import math
from preprocess import clean_str
import matplotlib.pyplot as plt

sent_number = 0
word_number = 0
words_list = []
max_sent_len = 0
max_sent = []
max_test_sent_len = 0
number_posts = 0
max_post_len = 0
max_post = []
sent_dict = {}
unknown = 0


with open('train.csv') as test:
    next(test)
    r = csv.reader(test, delimiter=',')
    for l in r:
        number_posts += 1

        if l[1] not in sent_dict:
            sent_dict[l[1]] = 0
        else:
            sent_dict[l[1]] += 1

        #l[3] = clean_str(l[3])

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

#find the Standard Deviation
with open('train.csv') as test:
    r = csv.reader(test, delimiter=',')
    next(test)
    total_dev = 0
    for l in r:
        #l[3] = clean_str(l[3])
        total_dev += len(l[3].split()) - mean
    standard_deviation = int(math.sqrt(total_dev ** 2 / sent_number))

with open('test_for_you_guys.csv') as test:
    r = csv.reader(test, delimiter=',')
    next(test)
    unique_words = set(words_list)
    for l in r:
        #l[2] = clean_str(l[2])
        words = l[2].split()
        if len(words) > max_test_sent_len:
            max_test_sent_len = len(words)

        for word in words:
            if word not in unique_words:
                unknown += 1



if __name__ == "__main__":
    print("Number of Posts:", number_posts)
    print("Maximum Post Length", max_post_len)
    print("The post:\n", max_post)
    print("Number of Sentences:", sent_number)
    print("Maximum Sentence Length:", max_sent_len)
    print("The sentence:\n",max_sent)
    print("Average Sentence Length:", mean)
    print("Sentence Standard Deviation:", standard_deviation)
    print("10 most common words:\n", collections.Counter(words_list).most_common(10))
    print("Number of unique words:", len(set(words_list)))
    print("Number of words:", word_number)
    print("UNKNOWN rate:", unknown)
    print("Maximum Sentence Length in test set:", max_test_sent_len)
    print()
    for sent in sent_dict:
        print("Number examples for {}: {} ({}%)".format(sent, sent_dict[sent], sent_dict[sent]/20000 * 100))


    def get_cmap(n, name='hsv'):
        return plt.cm.get_cmap(name, n)

    cmap = get_cmap(13)
    plt.pie([sent_dict[x] for x in sent_dict], labels=[x for x in sent_dict], colors=[cmap(x) for x in range(13)], explode=([0.25]*13), startangle=90, autopct='%1.1f%%')
    plt.show()





