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

name = 'Classification_linear'
dataset_dir = 'C:\\Users\FutoshiTsutsumi\Desktop\python_test\\tensorflow\dataset_sample\pickle'
save_file = dataset_dir + '\{}.pkl'.format(name)

with open(save_file, 'rb') as f:
    dataset = pickle.load(f)
    group1,group2,t_group1,t_group2,X,T = dataset

#==========================================================

import tensorflow as tf

holder_X = tf.placeholder(tf.float32)
holder_T = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_normal([2,2],mean=0.0, stddev=1.0))
B = tf.Variable(tf.random_normal([1,2],mean=0.0, stddev=1.0))
Y = tf.nn.softmax(tf.matmul(holder_X,W) + B)
loss = - tf.reduce_sum(tf.log(Y) * holder_T, axis=0)

# loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=holder_T, logits=Y)

optimizer = tf.train.GradientDescentOptimizer(0.001)
train = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(501):
        # if step % 100 == 0:
        #     print((step, sess.run(loss, feed_dict={holder_X: X,holder_T: T})))
        sess.run(train, feed_dict={holder_X: X,holder_T: T})

    weight = sess.run(W)
    bias = sess.run(B)

# #==========================================================
import numpy as np

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

#----------------------------------------------------------
#plot_grid
grid_range = 10
resolution = 40
offset = 5

x1_grid = x2_grid = np.linspace(offset-grid_range, offset+grid_range, resolution)

xx, yy = np.meshgrid(x1_grid, x2_grid)
X_grid = np.c_[xx.ravel(), yy.ravel()]

#----------------------------------------------------------

Y = softmax(np.dot(X_grid,weight) + bias)
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

plt.plot(group1[:,0], group1[:,1], 'o')
plt.plot(group2[:,0], group2[:,1], 'o')

plt.plot(red_group[:,0],red_group[:,1],'o',alpha=0.5)
plt.plot(blue_group[:,0],blue_group[:,1],'o',alpha=0.5)

plt.show()
# #==========================================================