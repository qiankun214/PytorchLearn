
# 基于Pytorch的MLP实现
## 目标
- 使用pytorch构建MLP网络
- 训练集使用MNIST数据集
- 使用GPU加速运算
- 要求准确率能达到92%以上
- 保存模型
## 实现
### 数据集：MNIST数据集的载入
MNIST数据集是一种常用的数据集，为28\*28的手写数字训练集，label使用独热码，在pytorch中，可以使用`torchvision.datasets.MNIST()`和`torch.utils.data.DataLoader（）`来导入数据集,其中
- `torchvision.datasets.MNIST()`:用于下载，导入数据集
- `torch.utils.data.DataLoader（）`:用于将数据集整理成batch的形式并转换为可迭代对象


```python
import torch as pt
import torchvision as ptv
import numpy as np
```


```python
train_set = ptv.datasets.MNIST("../../pytorch_database/mnist/train",train=True,transform=ptv.transforms.ToTensor(),download=True)
test_set = ptv.datasets.MNIST("../../pytorch_database/mnist/test",train=False,transform=ptv.transforms.ToTensor(),download=True)
```


```python
train_dataset = pt.utils.data.DataLoader(train_set,batch_size=100)
test_dataset = pt.utils.data.DataLoader(test_set,batch_size=100)
```

### 网络结构构建
网络使用最简单的MLP模型，使用最简单的线性层即可构建,本次网络一共有3层全连接层，分别为28\*28->512,512->128,128->10,除了输出层的激活函数使用softmax以外，其他均采用relu


```python
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
```

    MLP (
      (fc1): Linear (784 -> 512)
      (fc2): Linear (512 -> 128)
      (fc3): Linear (128 -> 10)
    )


### 代价函数，优化器和准确率检测
代价函数使用交叉熵函数，使用numpy计算准确率（pytorch中也有相关函数），优化器使用最简单的SGD


```python
# loss func and optim
optimizer = pt.optim.SGD(model.parameters(),lr=0.01,momentum=0.9)
lossfunc = pt.nn.CrossEntropyLoss().cuda()

# accuarcy
def AccuarcyCompute(pred,label):
    pred = pred.cpu().data.numpy()
    label = label.cpu().data.numpy()
#     print(pred.shape(),label.shape())
    test_np = (np.argmax(pred,1) == label)
    test_np = np.float32(test_np)
    return np.mean(test_np)

# test accuarcy
# print(AccuarcyCompute(
#     np.array([[1,10,6],[0,2,5]],dtype=np.float32),
#     np.array([[1,2,8],[1,2,5]],dtype=np.float32)))
```

### 训练网络
训练网络的步骤分为以下几步：
1. 初始化，清空网络内上一次训练得到的梯度
2. 载入数据为Variable，送入网络进行前向传播
3. 计算代价函数，并进行反向传播计算梯度
4. 调用优化器进行优化


```python
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

    0 : 0.9
    100 : 0.84
    200 : 0.82
    300 : 0.88
    400 : 0.9
    500 : 0.92
    0 : 0.93
    100 : 0.91
    200 : 0.9
    300 : 0.91
    400 : 0.9
    500 : 0.91
    0 : 0.93
    100 : 0.91
    200 : 0.94
    300 : 0.91
    400 : 0.93
    500 : 0.92
    0 : 0.96
    100 : 0.94
    200 : 0.95
    300 : 0.91
    400 : 0.93
    500 : 0.94


### 测试网络
使用使用测试集训练网络，直接计算结果并将计算准确率即可


```python
accuarcy_list = []
for i,(inputs,labels) in enumerate(test_dataset):
    inputs = pt.autograd.Variable(inputs).cuda()
    labels = pt.autograd.Variable(labels).cuda()
    outputs = model(inputs)
    accuarcy_list.append(AccuarcyCompute(outputs,labels))
print(sum(accuarcy_list) / len(accuarcy_list))
```

    0.936700002551


### 保存网络
pytorch提供了两种保存网络的方法，分别是保存参数和保存模型
- 保存参数：仅仅保存网络中的参数，不保存模型，在load的时候要预先定义模型
- 保存模型：保存全部参数与模型，load后直接使用


```python
# only save paramters
pt.save(model.state_dict(),"../../pytorch_model/mlp/params/mlp_params.pt")

# save model
pt.save(model,"../../pytorch_model/mlp/model/mlp_model.pt")
```

    /home/sky/virtualpython/pytorch0p2/lib/python3.5/site-packages/torch/serialization.py:147: UserWarning: Couldn't retrieve source code for container of type MLP. It won't be checked for correctness upon loading.
      "type " + obj.__name__ + ". It won't be checked "



```python
test_save_net = MLP().cuda()
test_save_net.load_state_dict(pt.load("../../pytorch_model/mlp/params/mlp_params.pt"))
accuarcy_list = []
for i,(inputs,labels) in enumerate(test_dataset):
    inputs = pt.autograd.Variable(inputs).cuda()
    labels = pt.autograd.Variable(labels).cuda()
    outputs = model(inputs)
    accuarcy_list.append(AccuarcyCompute(outputs,labels))
print(sum(accuarcy_list) / len(accuarcy_list))
```

    0.936700002551



```python
test_save_model = pt.load("../../pytorch_model/mlp/model/mlp_model.pt")
accuarcy_list = []
for i,(inputs,labels) in enumerate(test_dataset):
    inputs = pt.autograd.Variable(inputs).cuda()
    labels = pt.autograd.Variable(labels).cuda()
    outputs = model(inputs)
    accuarcy_list.append(AccuarcyCompute(outputs,labels))
print(sum(accuarcy_list) / len(accuarcy_list))
```

    0.936700002551


## 问题记录
### Variable转numpy的问题
Variable目前没查到转为numpy的方法，考虑Variable中的数据保存在一个`torch.Tensor`中，该Tensor为`Variable.data`，直接将其转为numpy即可
### GPU产生的转换问题
GPU上的Tensor不能直接转换为numpy，需要一个在CPU上的副本，因此可以先使用`Variable.cpu()`创建CPU副本，再使用`Variable.data.numpy()`方法
