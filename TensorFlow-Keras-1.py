# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 11:04:24 2019

@author: Yuan
"""

import tensorflow as tf
from tensorflow.keras import layers

print(tf.VERSION)
print(tf.keras.__version__)

model = tf.keras.Sequential()
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

layers.Dense(64, activation='sigmoid')
layers.Dense(64, activation=tf.sigmoid)
layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l1(0.01))
layers.Dense(64, bias_regularizer=tf.keras.regularizers.l2(0.01))
layers.Dense(64, kernel_initializer='orthogonal')
layers.Dense(64, bias_initializer=tf.keras.initializers.constant(2.0))

model = tf.keras.Sequential([
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
        ])

model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss = 'categorical_crossentropy',
              metrics=['accuracy'])


import numpy as np
data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))

val_data = np.random.random((100, 32))
val_labels = np.random.random((100, 10))

model.fit(data, labels, epochs=10, batch_size=32,
          validation_data=(val_data, val_labels))


# Instantiates a toy dataset instance:
dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(32)
dataset = dataset.repeat()

# Don't forget to specify `steps_per_epoch` when calling `fit` on a dataset.
model.fit(dataset, epochs=10, steps_per_epoch=30)

dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(32).repeat()

val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
val_dataset = val_dataset.batch(32).repeat()

model.fit(dataset, epochs=10, steps_per_epoch=30,
          validation_data=val_dataset,
          validation_steps=3)

data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))

model.evaluate(data, labels, batch_size=32)
model.evaluate(dataset, steps=30)

result = model.predict(data, batch_size=32)
print(result.shape)


#用函数式API构建高级模型
inputs = tf.keras.Input(shape=(32, ))
x = layers.Dense(64, activation='relu')(inputs)
x = layers.Dense(64, activation='relu')(x)
predictions = layers.Dense(10, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=predictions)

model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(data, labels, batch_size=32, epochs=5)

#自定义层
class MyLayer(layers.Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)
    def build(self, input_shape):
        shape = tf.TensorShape((input_shape[1], self.output_dim))
        self.kernel = self.add_weight(name = 'kernel',
                                      shape = shape,
                                      initializer = 'uniform',
                                      trainable = True)
        super(MyLayer, self).build(input_shape)
        
    def call(self, inputs):
        return tf.matmul(inputs, self.kernel)
    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.output_dim
        return tf.TensorShape(shape)
    def get_config(self):
        base_config = super(MyLayer, self).get_config()
        base_config['output_dim'] = self.output_dim
        return base_config
    def from_config(cls, config):
        return cls(**config)
    
model = tf.keras.Sequential([
        MyLayer(10),
        layers.Activation('softmax')
        ])    

model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
              loss = 'categorical_crossentropy',
              metrics=['accuracy'])
model.fit(data, labels, batch_size=32, epochs=15)

#回调
callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
        tf.keras.callbacks.TensorBoard(log_dir='./logs')
        ]

model.fit(data, labels, batch_size=32, epochs=5, callbacks=callbacks,
          validation_data=(val_data, val_labels))

#保存和恢复
model = tf.keras.Sequential([
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
        ])
model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss = 'categorical_crossentropy',
              metrics=['accuracy'])

model.save_weights('./weights/my_model')

model.load_weights('./weights/my_model')









