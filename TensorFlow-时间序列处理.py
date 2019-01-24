# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 13:32:41 2019

@author: Yuan
"""

from __future__ import print_function
import numpy as np
import matplotlib 
matplotlib.use('agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib.timeseries.python.timeseries import NumpyReader

x = np.array(range(1000))
noise = np.random.uniform(-0.2, 0.2, 1000)
y = np.sin(np.pi * x / 100) + x / 200. + noise
plt.plot(x, y)
plt.savefig('timeseries_y.jpg')
