# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 09:55:35 2019

@author: Yuan
"""


import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import seaborn as sns

sns.set_context(rc={'figure.figsize': (14, 7) } )
figzize_me = figsize =(14, 7)

import pandas as pd
data_train = pd.read_csv("./data/titanic/train.csv")
data_train.info()

data_train.groupby('Survived').count()
data_train.head(3)

def set_missing_ages(p_df):
    p_df.loc[(p_df.Age.isnull()),'Age'] = p_df.Age.dropna().mean()
    return p_df
df = set_missing_ages(data_train)

import sklearn.preprocessing as preprocessing
scaler = preprocessing.StandardScaler()
df['Age_scaled'] = scaler.fit_transform(data_train['Age'].values.reshape(-1,1))
df['Fare_scaled'] = scaler.fit_transform(data_train['Fare'].values.reshape(-1,1))

def set_cabin_type(p_df):
    p_df.loc[(p_df.Cabin.notnull()),'Cabin'] = "Yes"
    p_df.loc[(p_df.Cabin.isnull()),'Cabin'] = "No"
    return p_df
df = set_cabin_type(df)

dummies_pclass = pd.get_dummies(data_train['Pclass'], prefix='Pclass')
dummies_pclass.head(3)

dummies_embarked = pd.get_dummies(data_train['Embarked'], prefix='Embarked')
dummies_embarked.loc[61]

dummies_sex = pd.get_dummies(data_train['Sex'], prefix='Sex')
dummies_sex.head(3)


df = pd.concat([df, dummies_embarked, dummies_sex, dummies_pclass], axis=1)

# noinspection PyUnresolvedReferences
df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

# 选择哪些特征作为训练特征
train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
train_df.head(1)

from abupy import AbuML
train_np = train_df.as_matrix()
y = train_np[:, 0]
x = train_np[:, 1:]
titanic = AbuML(x, y, train_df)

titanic.estimator.logistic_classifier()
titanic.cross_val_accuracy_score()

#构造非线性特征
df['Child'] = (data_train['Age'] <= 10).astype(int)
df['Age*Age'] = data_train['Age'] * data_train['Age']
df['Age*Age_scaled'] = scaler.fit_transform(df['Age*Age'])

df['Age*Class'] = data_train['Age'] * data_train['Pclass']
df['Age*Class_scaled'] = scaler.fit_transform(df['Age*Class'].values.reshape(-1,1))

# filter加入新增的特征
train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*|Child|Age\*Class_.*')
train_df.head(1)

train_np = train_df.as_matrix()
y = train_np[:, 0]
x = train_np[:, 1:]
titanic = AbuML(x, y, train_df)
titanic.estimator.logistic_classifier()
titanic.cross_val_accuracy_score()

titanic.importances_coef_pd()

titanic.feature_selection()


#L2 
def loss_func(X, W, b, y):
    C = 2
    s = score(X, W, b)
    p = softmax(s)
    return -np.mean(cross_entropy(y, p)) + np.mean(np.dot(w.T, w)/C)

from abupy import AbuML
titanic = AbuML.create_test_more_fiter()
titanic.estimator.logistic_classifier()

titanic.plot_learning_curve()

#交叉验证
iris = AbuML.create_test_fiter()
iris.estimator.knn_classifier()

from abupy import KFold
kf = KFold(len(iris.y), n_folds=10, shuffle=True)

for train_index, test_index in kf:
    x_train, x_test = iris.x[train_index], iris.x[test_index]
    y_train, y_test = iris.y[train_index], iris.y[test_index]
    
x_train.shape, x_test.shape, y_train.shape, y_test.shape

#GridSearch
from abupy import ABuMLGrid

param_grid = dict(n_neighbors=range(1,31))
best_score_, best_params_ = iris.grid_search_common_clf(param_grid,
                                                        cv=10,
                                                        scoring='accuracy')
best_score_, best_params_

titanic.plot_confusion_matrices()


#波士顿房价预测
from sklearn import datasets
from abupy import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from abupy import AbuML

scikit_boston = datasets.load_boston()
x = scikit_boston.data
y = scikit_boston.target
df = pd.DataFrame(data=np.c_[x,y], columns=np.append(scikit_boston.feature_names, ['MEDV']))
df.head(1)

df.info()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_trian)
x_test = scaler.fit_transform(x_test)

df = pd.DataFrame(data=np.c_[x_train, y_train], columns=np.append(scikit_boston.feature_names, ['MEDV']))
boston = AbuML(x_train, y_train, df)
boston.estimator.polynomial_regression(degree=1)
reg = boston.fit()

y_pred = reg.predict(x_test)
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)

boston.estimator.polynomial_regression(degree=2)
reg = boston.fit()
y_pred = reg.predict(x_test)
r2_score(y_test, y_pred)

boston.estimator.polynomial_regression(degree=3)
reg = boston.fit()
y_pred = reg.predict(x_test)
r2_score(y_test, y_pred)

#回归预测年龄
import pandas as pd
data_train = pd.read_csv("./data/titanic/train.csv")
data_train.info()

import seaborn as sns
sns.distplot(data_train["Age"].dropna(), kde=True, hist=True)

def set_missing_ages(p_df):
    p_df.loc[(p_df.Age.isnull()), 'Age'] = data_train.Age.dropna().mean()
    return p_df

data_train = set_missing_ages(data_train)
data_train_fix1 = set_missing_ages(data_train)
sns.distplot(data_train_fix1["Age"], kde-True, hist=True)

from abupy import AbuML
import sklearn.preprocessing as preprocessing

def set_missing_age2(p_df):
    age_df = p_df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    scaler = preprocessing.StandardScaler()
    age_df['Fare_scaled'] = scaler.fit_transform(age_df.Fare.values.reshape(-1,1))
    del age_df['Fare']
    
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()
    y_inner = known_age[:,0]
    x_inner = known_age[:,1:]
    
    rfr_inner = AbuML(x_inner, y_inner, age_df.Age.notnull())
    rfr_inner.estimator.polynomial_regression(degree=1)
    reg_inner = rfr_inner.fit()
    
    predicted_ages = reg_inner.predict(unknown_age[:, 1::])
    p_df.loc[(p_df.Age.isnull()), 'Age'] = predicted_ages
    return p_df

data_train = pd.read_csv('./data/titannic/train.csv')
data_train_fix2 = set_missing_ages2(data_train)
sns.distplot(data_train_fix2["Age"], kde=True, hist=True)

import numpy as np
def entropy(P):
    return -np.sum(P * np.log2(P))















