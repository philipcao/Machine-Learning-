# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 13:49:25 2019

@author: Yuan
"""

from __future__ import print_function

import numpy as np
np.random.seed(1337)
from keras.datasets import mnist
from keras.models import Sequential
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils

nb_classes = 10
img_size = 28*28

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(y_train.shape[0], img_size).astype('float32')/255
X_test = X_test.reshape(y_test.shape[0], img_size).astype('float32')/255
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

X_train.shape, Y_train.shape

model = Sequential([
Dense(512, input_shape=(img_size,)),
Activation('relu'),
Dropout(0.2),
Dense(512, input_shape=(512,)),
Activation('relu'),
Dropout(0.2),
Dense(10, input_shape=(512,), activation='softmax'),
])

model.summary()


model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


batch_size = 128 # 梯度下降批数据量
epochs = 100 # 循环训练集次数

# 训练
model.fit(X_train, Y_train,
          batch_size=batch_size, epochs=epochs,
          verbose=1, validation_data=(X_test, Y_test))

# 测试
score = model.evaluate(X_test, Y_test, verbose=0)
print('accuracy: {}'.format(score[1]))




