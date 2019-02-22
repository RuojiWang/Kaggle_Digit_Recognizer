#coding=utf-8
#http://www.cs.toronto.edu/~kriz/cifar.html 这里有数据集的下载，顺便看看是否有mnist数据集下载吗？
#至少这个例子让我彻底熟悉了pytorch的流程咯。所以这个下午还是有收获的。晚上的时候准备研究一下提交的问题咯
#然后运行一下这个程序学点东西，然后准备看一下我现在的程序存在什么问题，最后提交一次就vans了吧。我现在感觉是系数的问题。
#我以前觉得大师和改进应该是任何方面的，但是八二定律也可以用在目标和原则上呀，只需要做好这些就行了吧。
#其实我一直是一个有目标和追求的人，只不过我追求的目标比较非主流吧，我现在要开始准备追求主流的目标了。
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

import sys
sys.path.append('D:/eclipse-SDK-4.5-win32/eclipse/plugins/org.python.pydev_4.5.1.201601132212/pysrc/pydevd.py')

# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1]
transform=transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                             ])
 
#训练集，将相对目录./data下的cifar-10-batches-py文件夹中的全部数据（50000张图片作为训练数据）加载到内存中，若download为True时，会自动从网上下载数据并解压
#trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
#NM$L 我之前一直不知道这个root的参数应该如何设置，我透尼玛我单步调试之后才知道如何设置的。我之前一直设置root='\cifar-10-batches-py\data_batch_1'
trainset = torchvision.datasets.CIFAR10(root='', train=True, download=False, transform=transform)
 
#将训练集的50000张图片划分成12500份，每份4张图，用于mini-batch输入。shffule=True在表示不同批次的数据遍历时，打乱顺序。num_workers=2表示使用两个子进程来加载数据
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, 
                                          shuffle=True, num_workers=2)
 
#测试集，将相对目录./data下的cifar-10-batches-py文件夹中的全部数据（10000张图片作为测试数据）加载到内存中，若download为True时，会自动从网上下载数据并解压
#testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
testset = torchvision.datasets.CIFAR10(root='', train=False, download=False, transform=transform)
 
#将测试集的10000张图片划分成2500份，每份4张图，用于mini-batch输入。
testloader = torch.utils.data.DataLoader(testset, batch_size=4, 
                                          shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
 
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) # 定义conv1函数的是图像卷积函数：输入为图像（3个频道，即彩色图）,输出为6张特征图, 卷积核为5x5正方形
        self.pool  = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)
 
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#果然在windows下面必须加上这个，不然程序就会出现奇怪的错误。
#出现的错误要么是 BrokenPipeError: [Errno 32] Broken pipe（直接运行）
#要么出现的错误是 ModuleNotFoundError: No module named 'pydevd' PermissionError: [WinError 5] 拒绝访问。（调试的时候）
#加上了__name__这个之后出现的错误是IndexError: invalid index of a 0-dim tensor.（直接运行）
#要么出现的错误是 ModuleNotFoundError: No module named 'pydevd' PermissionError: [WinError 5] 拒绝访问。（调试的时候）
#果然cnn_with_test1修改了这两个部分就可以运行咯，但是还是无法进行调试我不知道这是为什么呢？
if __name__ == '__main__':
    
    net = Net()
 
    criterion = nn.CrossEntropyLoss() #叉熵损失函数
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  #使用SGD（随机梯度下降）优化，学习率为0.001，动量为0.9
 
    for epoch in range(2): # 遍历数据集两次
     
        running_loss = 0.0
        #enumerate(sequence, [start=0])，i序号，data是数据
        for i, data in enumerate(trainloader, 0): 
        # get the inputs
            inputs, labels = data   #data的结构是：[4x3x32x32的张量,长度4的张量]
         
            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)  #把input数据从tensor转为variable
         
            # zero the parameter gradients
            optimizer.zero_grad() #将参数的grad值初始化为0
         
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels) #将output和labels使用叉熵计算损失
            loss.backward() #反向传播
            optimizer.step() #用SGD更新参数
         
            #因为没有办法进行调试，所以采用这样的方式进行输出
            print(inputs)
            print(type(inputs))
            #下面的输出是：torch.Size([4, 3, 32, 32])，意思是4张图3通道32*32的大小，所以那个应该是系数的问题吧?
            print(inputs.size())
            print(labels)
            print(type(labels))
            print(labels.size())
            
            # 每2000批数据打印一次平均loss值
            #之前下面的这一行代码报错了，应该是pytorch版本的问题，应该要修改成下面的样子
            #running_loss += loss.data[0]  #loss本身为Variable类型，所以要使用data获取其Tensor，因为其为标量，所以取0
            running_loss += loss.data
            
            if i % 2000 == 1999: # 每2000批打印一次
                print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss / 2000))
                running_loss = 0.0
 
    print('Finished Training')
 
    correct = 0
    total = 0
    for data in testloader:
        images, labels = data
        outputs = net(Variable(images))
        #print outputs.data
        _, predicted = torch.max(outputs.data, 1)  #outputs.data是一个4x10张量，将每一行的最大的那一列的值和序号各自组成一个一维张量返回，第一个是值的张量，第二个是序号的张量。
        total += labels.size(0)
        correct += (predicted == labels).sum()   #两个一维张量逐行对比，相同的行记为1，不同的行记为0，再利用sum(),求总和，得到相同的个数。
 
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))