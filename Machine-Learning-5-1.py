# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 10:42:56 2019

@author: Yuan
"""

from __future__ import print_function
import numpy as np
np.random.seed(1337)
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

batch_size = 128
nb_classes = 10
epochs = 100
img_size = 28 * 28

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(y_train.shape[0], img_size).astype('float32') / 255
X_test = X_test.reshape(y_test.shape[0], img_size).astype('float32')/255

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential([
        Dense(128, input_shape=(img_size,), activation='relu'),
        Dense(10, input_shape=(128,), activation='softmax'),])

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, Y_train,
          batch_size=batch_size, epochs=epochs,
          verbose=1, validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test, verbose=0)
print('accuracy:{}'.format(score[1]))









