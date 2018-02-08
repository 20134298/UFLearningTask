import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from model import FasionMNIST_NET
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import torch.onnx
import shutil
import random
import DenseNet as dn
import numpy as np
import matplotlib.pyplot as plt

T0 = torch.zeros(1, 28, 28)
T1 = torch.zeros(1, 28, 28)
T2 = torch.zeros(1, 28, 28)
T3 = torch.zeros(1, 28, 28)
for i in range(0, 28):
    for j in range(0, 28):
        if i == j:
            T0[0, i, j] = 1
        if (i + j) == 27:
           T3[0, i, j] = 1
        if (i - j) == 1:
            T1[0, i, j] = 1
        if (j - i) == 1:
            T2[0, i, j] = 1

trans_set = [T0, T1, T2, T3]

writer = SummaryWriter('log')

best_acc = 0

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                             transform=transform,
                                             target_transform=None,
                                             download=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=2)
testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                            transform=transform,
                                            target_transform=None,
                                            download=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False, num_workers=2)

classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress',
           'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')

net = FasionMNIST_NET()

#net = dn.DenseNet3(40, 10, 12, reduction=1.0, bottleneck=None, dropRate=0)
if torch.cuda.is_available():
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))

# dumy_input = Variable(torch.randn(128, 3, 32, 32)).cuda()
# torch.onnx.export(net, dumy_input, 'resnet.proto', verbose=True)
# writer.add_graph_onnx('resnet.proto')

criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
optimizer = optim .Adam(net.parameters(), lr=5e-3, betas=(0.9, 0.999), weight_decay=1e-3)

def test():
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(testloader):
        if torch.cuda.is_available():
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        print(batch_idx, len(testloader),
              'Text ',
              ' Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

def save_checkpoint(state, is_best, filename='model_save\\checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_save\\model_best.pth.tar')


if __name__ == '__main__':
    name_res = 'model_save\\res_checkpoint.pth.tar'
    res_checkpoint = torch.load(name_res)
    net.load_state_dict(res_checkpoint['state_dict'])
    print("=> loaded checkpoint or ResNet")

    test()
    testdata_set = iter(testloader)
    Test, label = testdata_set.next()

    if torch.cuda.is_available():
        inputs, targets = Test.cuda(), label.cuda()
    inputs, targets = Variable(inputs, volatile=True), Variable(targets)
    outputs = net(inputs)
    imgs = Test.numpy()
    _, predicted = torch.max(outputs.data, 1)
    for i in range(128):
        plt.imshow(imgs[i].reshape(28, 28), cmap='gray')
        print('Label is a', classes[label[i]])
        print('The network detected a', classes[predicted[i]])
        plt.show()

    writer.close()
