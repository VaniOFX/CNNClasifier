import torch.nn as nn
import torch.nn.functional as F
from DataSet import word2idx, sent_dict
import torch.optim as optim


LEARNING_RATE = 0.0003
EMBEDDING_DIM = 100
DROPOUT_PROB = 0.5

class CNN(nn.Module):

    def __init__(self, vocab_size, embedding_dim, sent_class, dropout_prob):
        super(CNN, self).__init__()
        self.emb = nn.Embedding(vocab_size, embedding_dim).cuda()
        self.conv1 = nn.Conv1d(1, 10, kernel_size=5*embedding_dim, stride=1)
        self.mp = nn.MaxPool1d(3)
        self.linear = nn.Linear(4330, sent_class)
        self.dropout = nn.Dropout(p=dropout_prob)


    def forward(self, inp):
        out = self.emb(inp)
        out = out.view(out.size(0), 1, -1)
        out = self.mp(F.relu(self.conv1(out)))
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.linear(out)
        out = F.log_softmax(out, dim=1)
        return out


model = CNN(len(word2idx), EMBEDDING_DIM, len(sent_dict), DROPOUT_PROB).cuda()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
