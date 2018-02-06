'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        #self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(256*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # print(x.size(),'0')
        out = F.relu(self.bn1(self.conv1(x)))
        # print(out.size(),'1')
        out = self.layer1(out)
        # print(out.size(),'2')
        out = self.layer2(out)
        # print(out.size(),'3')
        out = self.layer3(out)
        # print(out.size(),'4')
        #out = self.layer4(out)
        out = F.max_pool2d(out, 3)
        # print(out.size(),'5')
        out = out.view(out.size(0), -1)
        # print(out.size(),'6')
        out = self.linear(out)
        # print(out.size(), '7')
        return out


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.dropout2D = nn.Dropout2d(0.3)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(1, 8, 3)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.bn2 = nn.BatchNorm2d(16)

        self.fc1 = nn.Linear(16 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        # print(x.size(), '1')
        x = self.pool(F.relu(self.conv1(x)))
        x = self.bn1(x)
        # print(x.size(), '2')
        #x = self.dropout1(x)
        # print(x.size(), '3')
        x = self.pool(F.relu(self.conv2(x)))
        # print(x.size(), '4')
        #x = self.dropout1(x)
        # print(x.size(), '5')
        x = self.bn2(x)
        x = x.view(-1, 16 * 5 * 5)
        # print(x.size(), '6')
        x = F.relu(self.fc1(x))
        # print(x.size(), '7')
        x = F.relu(self.fc2(x))
        # print(x.size(), '8')
        x = self.fc3(x)
        x = F.softmax(x)
        # print(x.size(), '9')
        return x

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.dropout2D = nn.Dropout2d(0.3)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(1, 48, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        self.conv2 = nn.Conv2d(48, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 192, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(192)
        self.conv4 = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(192)
        self.conv5 = nn.Conv2d(192, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128 * 7 * 7, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        # print(x.size(), '1')

        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        # print(x.size(), '2')

        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        # print(x.size(), '3')

        x = self.pool(x)
        # print(x.size(), 'pool')

        x = F.relu(self.conv4(x))
        x = self.bn4(x)
        # print(x.size(), '4')

        x = F.relu(self.conv5(x))
        x = self.bn5(x)
        # print(x.size(), '5')

        x = self.pool(x)
        # print(x.size(), 'pool')

        x = x.view(-1, 128 * 7 * 7)
        # print(x.size(), 'view')
        x = F.relu(self.fc1(x))
        # print(x.size(), 'fc1')
        x = F.relu(self.fc2(x))
        # print(x.size(), 'fc2')
        x = self.fc3(x)
        # print(x.size(), 'fc3')
        x = F.softmax(x)
        # print(x.size(), '9')
        return x


def FasionMNIST_NET():
    return AlexNet()
    #return CNN()
    #return Net()
    #return ResNet(BasicBlock, [3,3,3])
    pass
