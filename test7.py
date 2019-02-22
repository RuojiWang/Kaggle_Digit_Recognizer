#coding=utf-8
#这个比赛的好处应该是不太需要做很多的特征工程吧
#这周之内rush掉数字识别吧，下周rush掉波士顿房价预测
#https://zhuanlan.zhihu.com/p/37613780
#https://github.com/IEavan/Kaggle-MNIST/blob/master/train.py
#https://github.com/yrowe/Digit-Recognizer-with-pytorch/blob/master/main-gpu.py 个人感觉左侧这个才是最正的解法
#https://www.jianshu.com/p/e4c7b3eb8f3d 这个是最简单的入门教程咯，且行且珍惜吧。
#关于目标的问题，以前只是想随心所欲的实现软件，也取得了一些阶段性的成果。主要体现为自助思考的一系列算法、知识体系
#但是当前存在的问题也很突出：1）视野眼界明显不够有点掉队了，掉队牺牲发展来维护自己的自尊心和研究，自己的研究这种有点低水平重复造轮子；
#2）现在的目标还不是很清晰，今天看到达摩院的目标才发现行业影响力等，才是现在今天职场生涯追求的关键问题吧。
#1）我一直就不擅长交流和人际吧，2）很多问题缺乏经验和见识在低水平重复，3）一直闭门造车走自己的老路子，4）不追求现实的价值和目标过多追寻兴趣和感受。
#以上都只是之前的问题，现在的新问题还在积累：1）自己的发展趋于滞后已经被同龄人反超，这个问题之后两年应该会爆发。2）到目前为止没有具体可以执行的计划和目标。
#不过稍微好点的地方就是自己的眼界更大了，有点明白了信仰加持和自己生活中的偶像吧
#现在这个项目的任务就是选择一个合适的结构，然后提交一次我的预测结果就行了吧。。
import os
import sys
import random
import pickle
import datetime
import warnings
import numpy as np
import pandas as pd

#原来DictVectorizer类也可以实现OneHotEncoder()的效果，而且更简单一些
from sklearn.cross_validation import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler

import torch
import torch.nn.init
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier, IsolationForest

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.cross_validation import train_test_split

from sklearn.model_selection import KFold, RandomizedSearchCV

import skorch
from skorch import NeuralNetClassifier

from sklearn import svm
from sklearn.covariance import EmpiricalCovariance, MinCovDet

import hyperopt
from hyperopt import fmin, tpe, hp, space_eval, rand, Trials, partial, STATUS_OK

from tpot import TPOTClassifier

from xgboost import XGBClassifier

from mlxtend.classifier import StackingCVClassifier

from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model.logistic import LogisticRegressionCV
from nltk.classify.svm import SvmClassifier
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.neural_network.multilayer_perceptron import MLPClassifier

#加载文件
data_train = pd.read_csv("train.csv")
data_test = pd.read_csv("test.csv")

#将文件转化为像个图片的样子
Y_train = data_train['label']
Y_train = pd.DataFrame(data=Y_train, columns=['label'])
X_train = data_train.drop(['label'], axis=1)
X_test = data_test

#将文件变为图片的样子
X_train = X_train/255
X_test = X_test/255
#print(X_train.values.shape) #(42000, 784)
#X_train = X_train.values.reshape(X_train.shape[0],28,28,1).astype('float32')
#这个是表示不改变行数的意思吧，将列数变为28x28x1，我在想是不是应该修改为1x28x28呀
X_train = X_train.values.reshape(X_train.shape[0],1,28,28).astype('float32') 
#print(X_train.shape) (42000, 28, 28, 1)
X_test = X_test.values.reshape(X_test.shape[0],1,28,28).astype('float32')

#为了能够训练他们，我将他们都转化为dataframe类型咯
#下面的写法会产生右侧错误RuntimeError: multi-target not supported at c:\pr
#Y_train = Y_train.values 只能够按照下面的写法进行的吧
Y_train = Y_train.values.reshape(Y_train.shape[0])

"""
#根据目前收集到的资料，感觉好像不太需要采用one-hot编码吧
#进行one-hot编码啦
#所以最后最简洁的也是我最原始的方式咯
label_mapping = {  
           1: '1',  
           2: '2',  
           3: '3',
           4: '4',
           5: '5',
           6: '6',
           7: '7',
           8: '8',
           9: '9',
           0: '0'}
Y_train['label'] = Y_train['label'].map(label_mapping)
dict_vector = DictVectorizer(sparse=False)
Y_train = dict_vector.fit_transform(Y_train.to_dict(orient='record'))
Y_train = pd.DataFrame(data=Y_train, columns=dict_vector.feature_names_)
"""

#https://blog.csdn.net/g11d111/article/details/82665265
#我感觉我好想没看懂下面的系数的设置呢？所以问题应该是出在这里吧？
class ClassifierModule(nn.Module):
    def __init__(self):
        super(ClassifierModule, self).__init__()

        self.cnn = nn.Sequential(
            #这下面几个参数分别是batch_size, 通道数， 图像的高宽像素
            #其实不用这么苦恼这几个参数的含义，可以定义一下然后调试看结果
            #调试结果为Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1))
            #所以我现在已经知道了所有参数的情况了，现在可以动手确定系数的问题了吧
            #https://zhuanlan.zhihu.com/p/43637179 按照这个解释，输出确实还是28x28呀
            nn.Conv2d(1, 32, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3, 3)),
            nn.ReLU(),
            #MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.25),
        )
        self.out = nn.Sequential(
            nn.Linear(64 * 12 * 12, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
            nn.Softmax(dim=-1),#最后都是设置-1为参数自动调整类别数目就行啦
        )

    def forward(self, X, **kwargs):
        X = self.cnn(X)
        X = X.reshape(-1, 64 * 12 * 12)
        X = self.out(X)
        return X

"""
#https://zhuanlan.zhihu.com/p/32190799 这里有个例子呢，将就这个例子修改一下呢
input=torch.ones(1,3,5,5)
input=Variable(input)
x=torch.nn.Conv2d(in_channels=3,out_channels=4,kernel_size=3,groups=1)
out=x(input)
print(list(x.parameters()))
"""
"""
#下面的这个例子证明了我之前的猜想，能够看到每一步的输出咯
input=torch.ones(10,1,28,28)
input=Variable(input)
x=nn.Conv2d(1, 32, (3, 3))
out=x(input) #torch.Size([10, 32, 26, 26])
print(list(x.parameters()))
x = nn.Conv2d(32, 64, (3, 3))
out=x(out) #torch.Size([10, 64, 24, 24])
x = nn.MaxPool2d((2, 2))
out=x(out) #torch.Size([10, 64, 12, 12])
"""

#调试结果为Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1))
#conv1 = nn.Conv2d(1, 32, (3, 3))
#调试结果为MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
#maxpool1 = nn.MaxPool2d((2, 2))

#现在遇到的新问题是这个RuntimeError: multi-target not supported at
#其实之前遇到过这个问题的就是Y_train的数据维度不对，reshape一下就完事了
#做了这个实验，居然cpu执行一个epoch花费了103.58s真是惊呆我了，GPU才行呀
#家里面的cpu居然也是90s一个epoch，gpu居然才1.6s一个epoch真是惊呆我啦
start_time = datetime.datetime.now()
clf = ClassifierModule()
net = NeuralNetClassifier(
        module = clf,
        batch_size=1024,
        optimizer=torch.optim.Adadelta,
        lr=0.0010,
        device="cuda",
        max_epochs=9600, #我的天600 1200 都完全不够，1200还用了41分钟还超参搜索尼玛呢。。
        callbacks = [skorch.callbacks.EarlyStopping(patience=10)],
    )
net.fit(X_train.astype(np.float32), Y_train.astype(np.longlong))

end_time = datetime.datetime.now()
print("time cost", (end_time - start_time))
