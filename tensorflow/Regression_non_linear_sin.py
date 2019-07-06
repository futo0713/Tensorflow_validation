import time
s = time.time()

#to turn off the warning 
#----------------------------------------------------------
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#----------------------------------------------------------

#inatall dataset
#==========================================================
import pickle

dataset_dir = 'C:\\Users\FutoshiTsutsumi\Desktop\python_test\\tensorflow\dataset_sample\pickle'
save_file = dataset_dir + '\Regression_non_linear_sin.pkl'

with open(save_file, 'rb') as f:
    x,t = pickle.load(f)

#==========================================================

import tensorflow as tf

holder_x = tf.placeholder(tf.float32)
holder_t = tf.placeholder(tf.float32)

Wh = tf.Variable(tf.random_normal([20,1],mean=0.0, stddev=1.0))
Bh = tf.Variable(tf.random_normal([20,1],mean=0.0, stddev=1.0))
H = tf.sigmoid(holder_x * Wh + Bh)

Wo = tf.Variable(tf.random_normal([1,20],mean=0.0, stddev=1.0))
Bo = tf.Variable(tf.random_normal([1],mean=0.0, stddev=1.0))
Y = tf.matmul(Wo, H) + Bo

loss = tf.reduce_mean(tf.square(Y - holder_t))

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(5001):
        if step % 500 == 0:
            print((step, sess.run(loss, feed_dict={holder_x: x,holder_t: t})))
        sess.run(train, feed_dict={holder_x: x,holder_t: t})

    weight_h = sess.run(Wh)
    bias_h = sess.run(Bh)
    weight_o = sess.run(Wo)
    bias_o = sess.run(Bo)

#Time required
e = time.time()
print('time:' + str(e-s))

#Plot graph
import numpy as np
import matplotlib.pyplot as plt

plt.plot(x,t,'.')

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

H = sigmoid(x*weight_h+bias_h)
Y = np.dot(weight_o,H) + bias_o

plt.plot(x,Y.reshape(-1),'.')
plt.ylim(0)
plt.show()
