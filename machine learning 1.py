# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 14:46:19 2019

@author: Yuan
"""

"""
import numpy as np # 快速操作结构数组的工具
import pandas as pd # 数据分析处理工具
import matplotlib.pyplot as plt # 画图工具
from sklearn import datasets # 机器学习库

#数据集 0-setosa、1-versicolor、2-virginica
scikit_iris = datasets.load_iris()
# 转换成pandas的DataFrame数据格式，方便观察数据
iris = pd.DataFrame(data=np.c_[scikit_iris['data'], scikit_iris['target']],
                     columns=np.append(scikit_iris.feature_names, ['y']))

iris.head(2)

iris.isnull().sum()

iris.groupby('y').count()

X = iris[scikit_iris.feature_names]
y = iris['y']

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X, y)
knn.predict([[3,2,2,5]])

from abupy import train_test_split
from sklearn import metrics
#
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=4)
#
knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(X_train, y_train)

y_pred_on_train = knn.predict(X_train)
y_pred_on_test = knn.predict(X_test)
#
print('accuracy: :{}'.format(metrics.accuracy_score(y_test, y_pred_on_test)))

from abupy import AbuML
#
iris = AbuML.create_test_fiter()
#
iris.estimator.knn_classifier(n_neighbors=15)
#
iris.cross_val_accuracy_score()
"""

#逻辑分类一：线性分类模型
import numpy as np
def score(x, w, b):
    return np.dot(x, w) + b

def sigmoid(s):
    return 1. / (1 + np.exp(-s))

def softmax(s):
    return np.exp(s) / np.sum(np.exp(s), axis=0)


import matplotlib.pyplot as plt
import seaborn as sns
x = np.arange(-3.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2*np.ones_like(x)])
plt.plot(x, softmax(scores).T, linewidth=2)

def cross_entropy(y,p):
    return np.sum(y * np.log(p) + (1 - y) * np.log(1 - p, axis=1))

def loss_func(X, w, b, y):
    s = score(X, w, b)
    y_p = softmax(s)
    return -np.mean(cross_entropy(y, y_p))

# 1.4 逻辑分类二：线性分类模型
import numpy as np
#
y = np.poly1d([1,0,0])
y(-7)

d_yx = np.polyder(y)
d_yx(-7)

import random
x_0 = random.uniform(-10, 10)
y_0 = random.uniform(-10, 10)
x_0, y_0

x = x_0
x_list = [x]
for i in range(10):
    x = step(x, d_yx)
    x_list.append(x)
x_list


def d_loss_func(X, w, b, y, w_i):
    #
    s = score(X, w, b)
    y_p = softmax(s)
    return np.mean(w_i * (y_p - y))

def d_b_loss(X, w, b, y, d_obj=1):
    #
    s = score(X, w, b)
    y_p = softmax(s)
    return np.mean(d_obj * (y_p - y))

def step(X, w, b, y, d_obj, loss_func):
    #
    alpha = .2
    return w_i - alpha * loss_func.__call__(X, w, b, y, d_obj)

class GDOptimizer:
    def optimize(X, y):
        #
        w1 = random.uniform(0,1)
        w2 = random.uniform(0,1)
        b = random.uniform(0,1)
        w = [w1, w2]
        #
        for i in range(100):
            w1 = step(X, w, b, y, w1, d_loss_func)
            w2 = step(X, w, b, y, w2, d_loss_func)
            b = step(X, w, b, y, b, d_b_loss)
            w = [w1, w2]
            
            
x = np.array([1, 2, 3, 4, 5])
assert np.mean(x) == np.sum(x) / 5

assert np.std(x) == np.sqrt(np.mean((x - np.mean(x))**2))

f1 = np.array([0.2, 0.5, 1.1]).reshape(-1,1)
f2 = np.array([-100.0, 56.0, -77.0]).reshape(-1,1)

f1_scaled = (f1 - np.mean(f1)) / np.std(f1)
f2_scaled = (f2 - np.mean(f2)) / np.std(f2)

import sklearn.preprocessing as preprocessing
scaler = preprocessing.StandardScaler()
f1_sk_scaled = scaler.fit_transform(f1)
f2_sk_scaled = scaler.fit_transform(f2)

assert np.allclose(f1_sk_scaled, f1_scaled) and np.allclose(f2_sk_scaled, f2_scaled)


from abupy import AbuML
import sklearn.preprocessing as preprocessing

iris = AbuML.create_test_fiter()

iris.estimator.logistic_classifier(multi_class='multinomial',solver='lbfgs')

iris.cross_val_accuracy_score()
