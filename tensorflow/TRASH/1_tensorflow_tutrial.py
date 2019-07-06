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

name = 'linear_class'
dataset_dir = 'C:\\Users\FutoshiTsutsumi\Desktop\python_test\\tensorflow\dataset_sample\pickle'
save_file = dataset_dir + '\dataset_{}.pkl'.format(name)

with open(save_file, 'rb') as f:
    dataset = pickle.load(f)
    X = dataset[0]
    T = dataset[1]
#==========================================================

import tensorflow as tf

holder_X = tf.placeholder(tf.float32)
holder_T = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_normal([2,2],mean=0.0, stddev=1.0))
B = tf.Variable(tf.random_normal([2,1],mean=0.0, stddev=1.0))

# Y = tf.matmul(W, holder_X) + B
# Y1 = tf.matmul(W, holder_X)

Y = tf.nn.softmax(tf.matmul(W, holder_X) + B)
Y1 = tf.nn.softmax(tf.matmul(W, holder_X))

# loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=holder_T, logits=Y)

# Wh = tf.Variable(tf.random_normal([20,1],mean=0.0, stddev=1.0))
# Bh = tf.Variable(tf.random_normal([20,1],mean=0.0, stddev=1.0))
# H = tf.sigmoid(holder_x * Wh + Bh)

# Wo = tf.Variable(tf.random_normal([1,20],mean=0.0, stddev=1.0))
# Bo = tf.Variable(tf.random_normal([1],mean=0.0, stddev=1.0))
# Y = tf.matmul(Wo, H) + Bo

loss = - tf.reduce_sum(tf.log(Y) * holder_T, axis=1)

optimizer = tf.train.GradientDescentOptimizer(0.001)
train = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # print(sess.run(holder_X,feed_dict={holder_X: X}).shape)
    # print(sess.run(W))
    # print(sess.run(B))
    print(sess.run(Y, feed_dict={holder_X: X}))
    print(sess.run(Y1, feed_dict={holder_X: X}))
    # print(sess.run(loss, feed_dict={holder_X: X, holder_T: T}))

    W = sess.run(W)
    B = sess.run(B)
    # Etf = sess.run(loss, feed_dict={holder_X: X, holder_T: T})

    # for step in range(3001):
    #     if step % 300 == 0:
    #         print((step, sess.run(loss, feed_dict={holder_X: X,holder_T: T})))
    #     sess.run(train, feed_dict={holder_X: X,holder_T: T})

    # print(sess.run(W))
    # print(sess.run(B))

    # weight = sess.run(W)
    # bias = sess.run(B)

import numpy as np

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

def loss(y, t):
    return -np.sum(np.multiply(t, np.log(y)) + np.multiply((1 - t), np.log(1 - y)))

Y = softmax(np.dot(W, X)+B)
Y1 = softmax(np.dot(W, X))
E = loss(Y, T)

print(W)
print(B)
print(Y)
print(Y1)

# print(-np.sum(np.log(Y)*T, axis=1))

# print(np.sum(-np.multiply(T,np.log(Y)),axis=1))

# print(E)
# print(Etf)

# print(weight)
# print(bias)


# #Time required
# e = time.time()
# print('time:' + str(e-s))

# #Plot graph
# import numpy as np
# import matplotlib.pyplot as plt

# plt.plot(x,t,'.')

# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

# H = sigmoid(x*weight_h+bias_h)
# Y = np.dot(weight_o,H) + bias_o

# plt.plot(x,Y.reshape(-1),'.')
# plt.ylim(0)
# plt.show()
