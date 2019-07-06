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

name = 'Classification_non_linear'
dataset_dir = 'C:\\Users\FutoshiTsutsumi\Desktop\python_test\\tensorflow\dataset_sample\pickle'
save_file = dataset_dir + '\{}.pkl'.format(name)

with open(save_file, 'rb') as f:
    dataset = pickle.load(f)
    X_center,X_circle,t_group1,t_group2,X,T = dataset

#==========================================================

import tensorflow as tf

holder_X = tf.placeholder(tf.float32)
holder_T = tf.placeholder(tf.float32)

Wh = tf.Variable(tf.random_normal([2,3],mean=0.0, stddev=1.0))
Bh = tf.Variable(tf.random_normal([1,3],mean=0.0, stddev=1.0))

Wo = tf.Variable(tf.random_normal([3,2],mean=0.0, stddev=1.0))
Bo = tf.Variable(tf.random_normal([1,2],mean=0.0, stddev=1.0))

H = tf.sigmoid(tf.matmul(holder_X, Wh) + Bh)
Y = tf.nn.softmax(tf.matmul(H,Wo) + Bo)
loss = - tf.reduce_sum(tf.log(Y) * holder_T, axis=0)

# loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=holder_T, logits=Y)

optimizer = tf.train.GradientDescentOptimizer(0.001)
train = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(5001):
        if step % 500 == 0:
            print((step, sess.run(loss, feed_dict={holder_X: X,holder_T: T})))
        sess.run(train, feed_dict={holder_X: X,holder_T: T})

    weight_h = sess.run(Wh)
    bias_h = sess.run(Bh)
    weight_o = sess.run(Wo)
    bias_o = sess.run(Bo)

#==========================================================
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

#----------------------------------------------------------
#plot_grid
grid_range = 10
resolution = 40
offset = 0

x1_grid = x2_grid = np.linspace(offset-grid_range, offset+grid_range, resolution)

xx, yy = np.meshgrid(x1_grid, x2_grid)
X_grid = np.c_[xx.ravel(), yy.ravel()]

#----------------------------------------------------------
H = sigmoid(np.dot(X_grid, weight_h) + bias_h)
Y = softmax(np.dot(H,weight_o) + bias_o)
Y_predict = np.hstack((X_grid,np.around(Y)))

red_group = Y_predict[Y_predict[:,2] == 1]
blue_group = Y_predict[Y_predict[:,3] == 1]

#==========================================================

#Time required
e = time.time()
print('time:' + str(e-s))

#==========================================================
#Plot graph
import matplotlib.pyplot as plt

plt.plot(X_center[:,0], X_center[:,1], 'o')
plt.plot(X_circle[:,0], X_circle[:,1], 'o')

plt.plot(red_group[:,0],red_group[:,1],'o',alpha=0.5)
plt.plot(blue_group[:,0],blue_group[:,1],'o',alpha=0.5)

plt.show()
#==========================================================