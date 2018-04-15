import csv
from preprocess import clean_str
from DataSet import word2idx, final_max
from torch.utils.data import DataLoader
import torch


def load_test_data(filename):

    with open(filename) as test:
        r = csv.reader(test, delimiter=',')
        next(test)
        data = []
        for l in r:
            temp = []
            unk = 0
            l[2] = clean_str(l[2])
            words = l[2].split()
            for word in words:
                if word in word2idx:
                    temp.append(word2idx[word])
                else:
                    unk += 1

            for i in range(final_max - len(words) + unk):
                temp.append(0)

            data.append(temp)

        data = torch.LongTensor(data)
        return DataLoader(dataset=data, batch_size=64)