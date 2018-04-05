import csv
import re
import collections
import math

sent_number = 0
word_number = 0
words_list = []
max_sent_len = 0
number_posts = 0
temp = []

with open('train.csv') as test:
    r = csv.reader(test, delimiter=',')
    for l in r:
        number_posts += 1

        #split the sentences in a post
        sents = re.split("[.;?!]+", l[3])

        #the number of sentences
        sent_number += len(sents)

        #the maximum sentence length
        for sent in sents:
            if len(sent.split()) > max_sent_len:
                max_sent_len = len(sent.split())
                temp = sent

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
        sents = re.split("[.;?!]+", l[3])
        total_dev += len(sent) - mean

    standard_deviation = int(math.sqrt(total_dev ** 2/ sent_number - 1))

#the 10 most common words

print("The most common words are\n", collections.Counter(words_list).most_common(10))
print("The number of sentences is ", sent_number)
print("The number of words is ", word_number)
print("The number of unique words is ", len(set(words_list)))
print("The maximum sentence length is", max_sent_len)
print(temp)
print("The average sentence length is", mean)
print("The sentence standart deviation is ", standard_deviation)





