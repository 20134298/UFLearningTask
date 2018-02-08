import torch
import torchvision
import torchvision.transforms as transforms
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
import matplotlib.pyplot as plt

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

if __name__ == '__main__':

    b = trainset[10][0].numpy()
    b_label = trainset[10][1]
    b = b.reshape(28, 28)
    plt.imshow(b, cmap='gray')
    print(classes[b_label])
    plt.show()