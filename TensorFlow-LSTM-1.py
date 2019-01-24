# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 13:31:36 2019

@author: Yuan
"""

"""
from sklearn import cross_validation
from sklearn import datasets
from sklearn import metrics
import tensorflow as tf

learn = tf.contrib.learn

def my_model(features, target):
    target = tf.one_hot(target, 3, 1, 0)
    logits, loss = learn.models.logistic_regression(features, target)
    train_op = tf.contrib.layers.optimize_loss(
            loss,
            tf.contrib.framework.get_global_step(),
            optimizer='Adagrad',
            learning_rate=0.1)
    return tf.arg_max(logits, 1), loss, train_op

iris = datasets.load_iris()
x_train, x_test, y_train, y_test = cross_validation.train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=0)

classifier = learn.Estimator(model_fn=my_model)
classifier.fit(x_train, y_train, steps=100)

y_predicted = classifier.predict(x_test)
score = metrics.accuracy_score(y_test, y_predicted)
print('Accuracy: %.2f%%' % (score*100))
"""
"""
import tensorflow as tf
input1 = tf.constant([1.0,2.0,3.0],name="input1")
input2 = tf.Variable(tf.random_uniform([3]), name="input2")
output = tf.add_n([input1,input2], name="add")

writer = tf.summary.FileWriter("/path/to/log", tf.get_default_graph())
writer.close()
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference

# 1. 定义神经网络的参数
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 3000
MOVING_AVERAGE_DECAY = 0.99

# 2. 定义训练的过程并保存TensorBoard的log文件
def train(mnist):
    #  输入数据的命名空间。
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = mnist_inference.inference(x, regularizer)
    global_step = tf.Variable(0, trainable=False)

    # 处理滑动平均的命名空间。
    with tf.name_scope("moving_average"):
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # 计算损失函数的命名空间。
    with tf.name_scope("loss_function"):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    # 定义学习率、优化方法及每一轮执行训练的操作的命名空间。
    with tf.name_scope("train_step"):
        learning_rate = tf.train.exponential_decay(
            LEARNING_RATE_BASE,
            global_step,
            mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY,
            staircase=True)

        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

        with tf.control_dependencies([train_step, variables_averages_op]):
            train_op = tf.no_op(name='train')

    # 训练模型。
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)

            if i % 1000 == 0:
                # 配置运行时需要记录的信息。
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                # 运行时记录运行信息的proto。
                run_metadata = tf.RunMetadata()
                _, loss_value, step = sess.run(
                    [train_op, loss, global_step], feed_dict={x: xs, y_: ys},
                    options=run_options, run_metadata=run_metadata)
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
            else:
                _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})

    writer = tf.summary.FileWriter("/path/to/log", tf.get_default_graph())
    writer.close()

# 3. 主函数
def main(argv=None):
    mnist = input_data.read_data_sets("../../datasets/MNIST_data", one_hot=True)
    train(mnist)

if __name__ == '__main__':
    main()

    







