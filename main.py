from torch.autograd import Variable
import torch
from DataSet import train_loader, test_loader, sent_dict
from CNNClasifier import model, optimizer, loss_function
from sklearn.metrics import classification_report



EPOCH = 50

def train():
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data).cuda(), Variable(target).cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print("the loss is", loss.data[0])


def test():
    test_loss = 0
    correct = 0
    predicted = torch.cuda.LongTensor()
    targets = torch.cuda.LongTensor()
    target_labels = [x for x in sent_dict]
    model.eval()
    for data, target in test_loader:
        data, target = Variable(data, volatile=True).cuda(), Variable(target).cuda()
        output = model(data)
        test_loss += loss_function(output, target).data[0]
        pred = output.data.max(1, keepdim=True)[1]


        predicted = torch.cat((predicted, pred))
        targets = torch.cat((targets, target.data.view_as(pred)))

        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    print(classification_report(targets, predicted, target_names=target_labels))
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

if __name__ == "__main__":
    for epoch in range(EPOCH):
        train()
        test()