# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 13:14:53 2019

@author: Yuan
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
print(tf.__version__)


imdb = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))


print(train_data[0])
print(len(train_data[0]), len(train_data[1]))

train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)



