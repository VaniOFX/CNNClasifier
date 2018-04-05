from torch.utils.data import Dataset, DataLoader
import csv
from preprocess import clean_str


bsz = 64
sent_dict = {}
word2idx = {}

class TwitterData(Dataset):
    
    def __init__(self, filename):
        X = []
        y_pred = []

        with open(filename) as test:
            r = csv.reader(test, delimiter=',')
            for l in r:
                #create the dictionary with the sentiment indeces
                if l[1] not in sent_dict:
                    sent_dict[l[1]] = len(sent_dict)

                #clean the sentence
                sentence = clean_str(l[3])

                #add to the lists
                X.append(sentence)
                y_pred.append(sent_dict[l[1]])

        self.x_data = X
        self.y_data = y_pred
        self.len = len(self.x_data)


    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


twitterData = TwitterData("train.csv")
# TODO split the data for into sets(sklearn)
train_loader = DataLoader(dataset=twitterData, batch_size=bsz, shuffle=True)