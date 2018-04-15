import torch.nn as nn
import torch.nn.functional as F
from DataSet import word2idx, sent2idx, max_post_len
import torch.optim as optim
from parameters import DROPOUT_PROB, EMBEDDING_DIM, LEARNING_RATE, KERNEL_SIZE, POOLING, STRIDE, K2, PADDING, POOLING_TYPE



class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.emb = nn.Embedding(len(word2idx), EMBEDDING_DIM).cuda()
        self.conv1 = nn.Conv1d(1, K2, kernel_size=KERNEL_SIZE, stride=STRIDE, padding=PADDING)
        if POOLING_TYPE == 'avg':
            self.pool = nn.AvgPool1d(POOLING)
        elif POOLING_TYPE == 'max':
            self.pool = nn.MaxPool1d(POOLING)
        self.linear1 = nn.Linear(
            int(((((max_post_len * EMBEDDING_DIM - KERNEL_SIZE + 2 * PADDING) / STRIDE) + 1) / POOLING)) * K2,
            256)
        self.linear2 = nn.Linear(256, len(sent2idx))
        self.dropout = nn.Dropout(p=DROPOUT_PROB)


    def forward(self, inp):
        out = self.emb(inp)
        out = out.view(out.size(0), 1, -1)
        out = self.conv1(out)
        out = F.relu(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = F.relu(self.linear1(out))
        out = self.dropout(out)
        out = self.linear2(out)
        out = F.log_softmax(out, dim=1)
        return out


model = CNN().cuda()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
