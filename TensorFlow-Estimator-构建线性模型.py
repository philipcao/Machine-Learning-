# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 08:46:07 2019

@author: Yuan
"""
"""
import tensorflow as tf

g1 = tf.Graph()
with g1.as_default():
    v = tf.get_variable(
            "v", initializer=tf.zeros_initializer()(shape=[1]))
g2 = tf.Graph()
with g2.as_default():
    v = tf.get_variable(
            "v", initializer=tf.ones_initializer()(shape=[1]))

with tf.Session(graph=g1) as sess:
    tf.initialize_all_variables().run()
    with tf.variable_scope("", reuse=True):
        print(sess.run(tf.get_variable("v")))

with tf.Session(graph=g2) as sess:
    tf.initialize_all_variables().run()
    with tf.variable_scope("", reuse=True):
        print(sess.run(tf.get_variable("v")))
        
"""
"""
import tensorflow as tf
w1 = tf.Variable(tf.random_normal([2,3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3,1], stddev=1, seed=1))

x = tf.constant([[0.7, 0.9]])

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

sess = tf.Session()

sess.run(w1.initializer)
sess.run(w2.initializer)
print(sess.run(y))
sess.close()
"""
"""
import tensorflow as tf
from numpy.random import RandomState

batch_size = 8

w1 = tf.Variable(tf.random_normal([2,3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3,1], stddev=1, seed=1))

x = tf.placeholder(tf.float32, shape=(None,2), name='x-input')
y_ = tf.placeholder(tf.float32, shape=(None,1), name='y-input')

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

cross_entropy = -tf.reduce_mean(
        y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)

rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
Y = [[int(x1+x2 < 1)] for (x1, x2) in X]

with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    print(sess.run(w1))
    print(sess.run(w2))
    
    STEPS = 5000
    for i in range(STEPS):
        start = (i * batch_size) % dataset_size
        end = min(start+batch_size, dataset_size)
        sess.run(
                train_step,
                feed_dict = {x: X[start:end], y_: Y[start:end]})
        if i % 1000 == 0:
            total_cross_entropy = sess.run(
                    cross_entropy, feed_dict = {x: X, y_: Y})
            print("After %d training step(s),cross entropy on all data is %g",
                  (i, total_cross_entropy))
            
    print (sess.run(w1))
    print (sess.run(w2))
sess.close()
"""





















