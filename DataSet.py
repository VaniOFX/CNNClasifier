from torch.utils.data import Dataset, DataLoader
import torch
import csv
from preprocess import clean_str
from statistics import max_post_len
from sklearn.model_selection import StratifiedShuffleSplit

bsz = 31
sent_dict = {}
word2idx = {'pad':0}


class TwitterData(Dataset):
    
    def __init__(self, filename):
        X = []
        y_pred = []

        with open(filename) as test:
            next(test)
            r = csv.reader(test, delimiter=',')
            for l in r:
                #create the dictionary with the sentiment indeces
                if l[1] not in sent_dict:
                    sent_dict[l[1]] = len(sent_dict)

                #clean the sentence
                sentence = clean_str(l[3])

                words = sentence.split()
                for word in words:
                    if word not in word2idx:
                        word2idx[word] = len(word2idx)

                sentence_idx = []
                for word in words:
                    sentence_idx.append(word2idx[word])

                for i in range(max_post_len-len(words)):
                    sentence_idx.append(0)

                #add to the lists
                X.append(sentence_idx)
                y_pred.append(sent_dict[l[1]])


        self.x_data = torch.LongTensor(X)
        self.y_data = torch.LongTensor(y_pred)
        self.len = len(self.x_data)


    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


twitterData = TwitterData("train.csv")
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
train_data = []
test_data = []
for train, test in sss.split(twitterData.x_data, twitterData.y_data):
    for i in train:
        train_data.append(twitterData[i])
    for i in test:
        test_data.append(twitterData[i])
train_loader = DataLoader(dataset=train_data, batch_size=bsz, shuffle=True)

