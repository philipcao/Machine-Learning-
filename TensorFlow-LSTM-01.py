# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 11:11:45 2019

@author: Yuan
"""

import tensorflow as tf
import numpy as np

rnn_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=128)
print(rnn_cell.state_size)

lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=128)
print(lstm_cell.state_size)

inputs = tf.placeholder(np.float32, shape=(32, 100))
h0 = lstm_cell.zero_state(32, np.float32)
output, h1 = lstm_cell(inputs, h0)

print(h1.h)
print(h1.c)

