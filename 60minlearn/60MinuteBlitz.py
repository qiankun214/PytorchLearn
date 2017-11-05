import torch as pt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
# Tensors
x = pt.Tensor(2, 4)
# print(x)
x = pt.rand(5, 3)
# print(x,x.size()[0])
y = pt.rand(5, 3)

# Operations
result = pt.Tensor(5, 3)
test = pt.add(x, y, out=result)
# print(result,test)

y.add_(x)
# print(y)
# print(y[1:4,:2])

# numpy
a = pt.ones(5)
b = a.numpy()
# print(a,b)
a.add_(1)
# print(a,b)

a = np.ones(5)
b = pt.from_numpy(a)
print(a, b)
np.add(a, 1, out=a)
# print(a,b)

# cuda
# print(pt.cuda.is_available())
a, b = pt.Tensor(2, 2), pt.Tensor(2, 2)
a = a.cuda()
b = b.cuda()
# print(a + b)

# variable
x = Variable(pt.ones(2, 2), requires_grad=True)
y = x + 2
z = y * y * 3
out = z.mean()
print(x, y, z, out)

# backward
out.backward()
print(x.grad)


class Net(nn.Module):
    """docstring for Net"""

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)  # input channel,6 output channel,5x5
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # input 16*5*5,output120
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))  # input,poolcore shape
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # (2,2) => 2 because 2=2
        x = x.view(-1, self.num_flat_features(x))  # reshape
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # remove batch size
        num_f = 1
        for s in size:
            num_f *= s
        return num_f

net = Net().cuda()
print(net)

net.zero_grad()  # Zero the gradient buffers of all parameters

# ?(1,1,32,32) nSamples x nChannels x Height x Width
inputdata = Variable(pt.randn(1, 1, 32, 32))
inputdata = inputdata.cuda()
# 1-batch 1-input channel 32,32
# print(inputdata)
# print(inputdata.unsqueeze(0)) #[torch.FloatTensor of size 1x1x1x32x32]
out = net(inputdata)
print(out)

# loss function
target = Variable(pt.arange(1, 11)).cuda()
# print(target)
criterion = nn.MSELoss().cuda()
loss = criterion(out, target)  # out shape = 1*10 target shape = 10
# print(loss)

# print(loss.grad_fn)

# backprop
net.zero_grad()
print(net.conv1.bias.grad)

loss.backward()
print(net.conv1.bias.grad)

# update weight
print(net.parameters())
for f in net.parameters():
    f.data.sub_(f.grad.data * 0.01)

optimizer = optim.SGD(net.parameters(), lr=0.01)
optimizer.zero_grad()
out = net(inputdata)
loss = criterion(out, target)
loss.backward()
optimizer.step()
