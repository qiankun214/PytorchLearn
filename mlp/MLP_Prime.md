
# MLP中实现dropout，批标准化
## 基本网络代码
- 三层MLP
- 使用MNIST数据集


```python
import torch as pt
import torchvision as ptv
import numpy as np

train_set = ptv.datasets.MNIST("../../pytorch_database/mnist/train",train=True,transform=ptv.transforms.ToTensor(),download=True)
test_set = ptv.datasets.MNIST("../../pytorch_database/mnist/test",train=False,transform=ptv.transforms.ToTensor(),download=True)
train_dataset = pt.utils.data.DataLoader(train_set,batch_size=100)
test_dataset = pt.utils.data.DataLoader(test_set,batch_size=100)

class MLP(pt.nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.fc1 = pt.nn.Linear(784,512)
        self.fc2 = pt.nn.Linear(512,128)
        self.fc3 = pt.nn.Linear(128,10)
        
    def forward(self,din):
        din = din.view(-1,28*28)
        dout = pt.nn.functional.relu(self.fc1(din))
        dout = pt.nn.functional.relu(self.fc2(dout))
        return pt.nn.functional.softmax(self.fc3(dout))
model = MLP().cuda()
print(model)

# loss func and optim
optimizer = pt.optim.SGD(model.parameters(),lr=0.01,momentum=0.9)
lossfunc = pt.nn.CrossEntropyLoss().cuda()

# accuarcy
def AccuarcyCompute(pred,label):
    pred = pred.cpu().data.numpy()
    label = label.cpu().data.numpy()
    test_np = (np.argmax(pred,1) == label)
    test_np = np.float32(test_np)
    return np.mean(test_np)

for x in range(4):
    for i,data in enumerate(train_dataset):
    
        optimizer.zero_grad()
    
        (inputs,labels) = data
        inputs = pt.autograd.Variable(inputs).cuda()
        labels = pt.autograd.Variable(labels).cuda()
    
        outputs = model(inputs)
    
        loss = lossfunc(outputs,labels)
        loss.backward()
    
        optimizer.step()
    
        if i % 100 == 0:
            print(i,":",AccuarcyCompute(outputs,labels))
```

    MLP (
      (fc1): Linear (784 -> 512)
      (fc2): Linear (512 -> 128)
      (fc3): Linear (128 -> 10)
    )
    0 : 0.1
    100 : 0.2
    200 : 0.34
    300 : 0.17
    400 : 0.51
    500 : 0.52
    0 : 0.79
    100 : 0.77
    200 : 0.69
    300 : 0.75
    400 : 0.85
    500 : 0.85
    0 : 0.88
    100 : 0.8
    200 : 0.76
    300 : 0.79
    400 : 0.85
    500 : 0.85
    0 : 0.89
    100 : 0.81
    200 : 0.77
    300 : 0.82
    400 : 0.85
    500 : 0.86


## 增加批标准化
批标准化是添加在激活函数之前，使用标准化的方式将输入处理到一个区域内或者近似平均的分布在一个区域内
在pytorch中，使用`torch.nn.BatchNorm1/2/3d（）`函数表示一个批标准化层，使用方法与其它层类似


```python
class MLP(pt.nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.fc1 = pt.nn.Linear(784,512)
        self.norm1 = pt.nn.BatchNorm1d(512,momentum=0.5)
        self.fc2 = pt.nn.Linear(512,128)
        self.norm2 = pt.nn.BatchNorm2d(128,momentum=0.5)
        self.fc3 = pt.nn.Linear(128,10)
        
    def forward(self,din):
        din = din.view(-1,28*28)
        dout = pt.nn.functional.relu(self.norm1(self.fc1(din)))
        dout = pt.nn.functional.relu(self.norm2(self.fc2(dout)))
        return pt.nn.functional.softmax(self.fc3(dout))
model_norm = MLP().cuda()
print(model_norm)

optimizer = pt.optim.SGD(model_norm.parameters(),lr=0.01,momentum=0.9)
lossfunc = pt.nn.CrossEntropyLoss().cuda()
```

    MLP (
      (fc1): Linear (784 -> 512)
      (norm1): BatchNorm1d(512, eps=1e-05, momentum=0.5, affine=True)
      (fc2): Linear (512 -> 128)
      (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.5, affine=True)
      (fc3): Linear (128 -> 10)
    )



```python
for x in range(6):
    for i,data in enumerate(train_dataset):
    
        optimizer.zero_grad()
    
        (inputs,labels) = data
        inputs = pt.autograd.Variable(inputs).cuda()
        labels = pt.autograd.Variable(labels).cuda()
    
        outputs = model_norm(inputs)
    
        loss = lossfunc(outputs,labels)
        loss.backward()
    
        optimizer.step()
    
        if i % 200 == 0:
            print(i,":",AccuarcyCompute(outputs,labels))
```

    0 : 0.2
    200 : 0.69
    400 : 0.89
    0 : 0.96
    200 : 0.95
    400 : 0.97
    0 : 0.97
    200 : 0.96
    400 : 0.99
    0 : 0.97
    200 : 0.97
    400 : 0.99
    0 : 0.97
    200 : 0.97
    400 : 0.99
    0 : 0.97
    200 : 0.98
    400 : 0.99



```python
accuarcy_list = []
for i,(inputs,labels) in enumerate(test_dataset):
    inputs = pt.autograd.Variable(inputs).cuda()
    labels = pt.autograd.Variable(labels).cuda()
    outputs = model_norm(inputs)
    accuarcy_list.append(AccuarcyCompute(outputs,labels))
print(sum(accuarcy_list) / len(accuarcy_list))
```

    0.976300007105


与不使用批标准化的网络（准确率93%左右）相比，使用批标准化的网络准确率由明显的提高

## dropout
dropout是一种常见的防止过拟合的方法，通过将网络中的神经元随机的置0来达到防止过拟合的目的
pytorch中使用`torch.nn.Dropout()`和`torch.nn.Dropout2/3d()`函数构造，且该层只在训练中起作用，在预测时dropout将不会工作


```python
class MLP(pt.nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.fc1 = pt.nn.Linear(784,512)
        self.drop1 = pt.nn.Dropout(0.6)
        self.fc2 = pt.nn.Linear(512,128)
        self.drop2 = pt.nn.Dropout(0.6)
        self.fc3 = pt.nn.Linear(128,10)
        
    def forward(self,din):
        din = din.view(-1,28*28)
        dout = pt.nn.functional.relu(self.drop1(self.fc1(din)))
        dout = pt.nn.functional.relu(self.drop2(self.fc2(dout)))
        return pt.nn.functional.softmax(self.fc3(dout))
model_drop = MLP().cuda()
print(model_drop)
optimizer = pt.optim.SGD(model_drop.parameters(),lr=0.01,momentum=0.9)
lossfunc = pt.nn.CrossEntropyLoss().cuda()
```

    MLP (
      (fc1): Linear (784 -> 512)
      (drop1): Dropout (p = 0.6)
      (fc2): Linear (512 -> 128)
      (drop2): Dropout (p = 0.6)
      (fc3): Linear (128 -> 10)
    )



```python
for x in range(6):
    for i,data in enumerate(train_dataset):
    
        optimizer.zero_grad()
    
        (inputs,labels) = data
        inputs = pt.autograd.Variable(inputs).cuda()
        labels = pt.autograd.Variable(labels).cuda()
    
        outputs = model_drop(inputs)
    
        loss = lossfunc(outputs,labels)
        loss.backward()
    
        optimizer.step()
    
        if i % 200 == 0:
            print(i,":",AccuarcyCompute(outputs,labels))
```

    0 : 0.11
    200 : 0.25
    400 : 0.32
    0 : 0.5
    200 : 0.51
    400 : 0.74
    0 : 0.8
    200 : 0.68
    400 : 0.8
    0 : 0.86
    200 : 0.74
    400 : 0.85
    0 : 0.88
    200 : 0.78
    400 : 0.8
    0 : 0.9
    200 : 0.75
    400 : 0.83



```python
accuarcy_list = []
for i,(inputs,labels) in enumerate(test_dataset):
    inputs = pt.autograd.Variable(inputs).cuda()
    labels = pt.autograd.Variable(labels).cuda()
    outputs = model_drop(inputs)
    accuarcy_list.append(AccuarcyCompute(outputs,labels))
print(sum(accuarcy_list) / len(accuarcy_list))
```

    0.840299996734


可以看到，dropout对于系统性能的还是有比较大的影响的，对于这种微型网络来说，泛化能力的提升并不明显

## 疑问
当批标准化和dropout同时存在时，这两个层次的相互位置该如何考虑
- -> dropout->norm->function?
- -> norm->dropout->function?
