import torch
import numpy
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision

train_data = torchvision.datasets.MNIST(
    './mnist', train=True, transform=torchvision.transforms.ToTensor(), download=True
)


class GeneratorNetwork(nn.Module):
    """docstring for GeneratorNetwork"""

    def __init__(self):
        super(GeneratorNetwork, self).__init__()
        self.fc0 = nn.Linear(10, 1024)
        self.deconv1 = nn.ConvTranspose2d(64, 32, 1, stride=2)
        self.deconv2 = nn.ConvTranspose2d(32, 8, 1, stride=2, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(8, 1, 1, stride=2, output_padding=1)

    def forward(self, x):
        x = F.relu(self.fc0(x))
        x = x.view(-1, 64, 4, 4)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.tanh(self.deconv3(x))
        return x


class ClassicNetwork(nn.Module):
    """docstring for ClassicNetwork"""

    def __init__(self):
        super(ClassicNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 4, 2)
        self.conv2 = nn.Conv2d(8, 24, 3, 2)
        self.conv3 = nn.Conv2d(24, 48, 2)
        self.fc1 = nn.Linear(48 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 2)

    def num_flat_features(self, x):
        size = x.size()[1:]  # remove batch size
        num_f = 1
        for s in size:
            num_f *= s
        return num_f

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x))
        return x


class GANNetwork(nn.Module):
    """docstring for GANNetwork"""

    def __init__(self):
        super(GANNetwork, self).__init__()
        self.G_network = GeneratorNetwork().cuda()
        self.D_network = ClassicNetwork().cuda()
        self.LossFunc = nn.CrossEntropyLoss()
        self.G_optimizer = torch.optim.Adam(
            self.G_network.parameters(), lr=0.0001, betas=0.9)
        self.D_optimizer = torch.optim.Adam(
            self.D_network.parameters(), lr=0.0001, betas=0.9)

    def G_LossFunc(self, D_output):
        return self.LossFunc(D_output, torch.ones(D_output.size()))

    def D_LossFunc(self, D_output_G, D_output_MNIST):
        G_part = self.LossFunc(D_output_G, torch.zeros(D_output_G.size()))
        MNIST_part = self.LossFunc(
            D_output_MNIST, torch.ones(D_output_MNIST.size()))
        return G_part + MNIST_part

    def TrainOneStep(self, x, rel_x):
        self.G_optimizer.zero_grad()
        self.D_optimizer.zero_grad()

        G_result = self.G_network.forward(x)
        G_classic = self.D_network.forward(G_result)
        MNIST_classic = self.D_network.forward(rel_x)

        G_loss = self.G_LossFunc(G_classic)
        D_loss = self.D_LossFunc(G_classic, MNIST_classic)

        G_loss.backward()
        D_loss.backward()

        self.G_optimizer.step()
        self.D_optimizer.step()

if __name__ == '__main__':
    # testg = GeneratorNetwork().cuda()
    # testc = ClassicNetwork().cuda()
    # x = Variable(torch.ones(1, 10), requires_grad=True).cuda()
    # y = testg.forward(x)
    # print(y)
    # y = testc.forward(y)
    # print(y.size())
    test = GANNetwork()
