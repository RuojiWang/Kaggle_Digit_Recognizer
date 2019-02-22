#coding=utf-8
#所以这个版本的工作就是准备找一些比较合适的结构吧
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

X_train = X_train.values.reshape(X_train.shape[0],1,28,28).astype('float32') 
X_test = X_test.values.reshape(X_test.shape[0],1,28,28).astype('float32')
Y_train = Y_train.values.reshape(Y_train.shape[0])

#https://blog.csdn.net/g11d111/article/details/82665265
#我感觉我好想没看懂下面的系数的设置呢？所以问题应该是出在这里吧？
class ClassifierModule(nn.Module):
    def __init__(self):
        super(ClassifierModule, self).__init__()

        self.cnn = nn.Sequential(

            #调试结果为Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1))
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

#现在需要做一下划分验证集和训练集，然后对未知数据进行预测和输出
#不对吧，如果进行提交不需要进行split的吧，但是需要进行输出
#这个版本算是能够进行计算和输出的版本了吧，但是如何加上超参搜索呢
start_time = datetime.datetime.now()

#X_split_train, X_split_test, Y_split_train, Y_split_test = train_test_split(X_train, Y_train, test_size=0.10, stratify=Y_train)

#我家里的GPU还是1.6s一个epoch，网上找到的一些教程居然是28s一个epoch，例如下面的教程
#https://www.kaggle.com/adityaecdrid/mnist-with-keras-for-beginners-99457
clf = ClassifierModule()
net = NeuralNetClassifier(
        module = clf,
        batch_size=1024,
        optimizer=torch.optim.Adadelta,
        lr=0.0010,
        device="cpu",
        max_epochs=20000, #我的天600 1200 都完全不够，1200还用了41分钟还超参搜索尼玛呢。。
        callbacks = [skorch.callbacks.EarlyStopping(patience=10)],
    )
net.fit(X_train.astype(np.float32), Y_train.astype(np.longlong))

Y_pred = net.predict(X_test)

data = {"ImageId":list(range(1,len(Y_pred)+1)), "Label":Y_pred}
output = pd.DataFrame(data = data)
output.to_csv("Digit_Recognizer_Prediction.csv", index=False)

end_time = datetime.datetime.now()
print("time cost", (end_time - start_time))
