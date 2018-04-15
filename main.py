from torch.autograd import Variable
import torch
from sklearn.metrics import classification_report, precision_recall_fscore_support
from parameters import EPOCH, bow, cnn, bsz
import matplotlib.pyplot as plt
from test_file_preprocess import load_test_data
from DataSet import sent2idx, idx2sent

if cnn:
    from CNNClasifier import model, optimizer, loss_function
else:
    from NNClasifier import model, optimizer, loss_function

if bow:
    from DataSetBOW import train_loader, test_loader
else:
    from DataSet import train_loader, test_loader

iterations = []
f_scores = []
losses = []
iter_loss = []

def train():
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data).cuda(), Variable(target).cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.data[0]

    final_loss = total_loss / 20000
    print("The final loss is {}".format(final_loss))
    losses.append(final_loss)
    iter_loss.append(len(iterations))


def test():
    predicted = torch.cuda.LongTensor()
    targets = torch.cuda.LongTensor()
    target_labels = [x for x in sent2idx]
    model.eval()
    for data, target in test_loader:
        data, target = Variable(data, volatile=True).cuda(), Variable(target).cuda()
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1]
        predicted = torch.cat((predicted, pred))
        targets = torch.cat((targets, target.data.view_as(pred)))

    print(classification_report(targets, predicted, target_names=target_labels))

    _, _, f, _ = precision_recall_fscore_support(targets, predicted, average='weighted')
    f_scores.append(f*100)
    iterations.append(len(iterations)+1)

def predict():
    predicted = torch.cuda.LongTensor()
    model.eval()
    predict_set = load_test_data('test_for_you_guys.csv')
    for data in predict_set:
        data = Variable(torch.LongTensor(data)).cuda()
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1]
        predicted = torch.cat((predicted, pred))
    write_predicted(predicted)


def write_predicted(predicted):
    f_pred = open('submission.csv', 'w')
    f_pred.write('ID\tSentiment')
    ID = 1
    for pred in predicted:
        f_pred.write(str(ID) + ',' + str(idx2sent[int(pred)]) + '\n')
        ID += 1



if __name__ == "__main__":
    for epoch in range(EPOCH):
        print("Epoch number {} starting get ready:".format(epoch+1))
        train()
        test()

    predict()

    plt.plot(iterations, f_scores)
    plt.xlabel('Iteration')
    plt.ylabel('F1 Score')
    plt.show()

    plt.plot(iter_loss, losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.show()