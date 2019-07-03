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

print('- What kind of data?')

name = input()
dataset_dir = 'C:\\Users\FutoshiTsutsumi\Desktop\python_test\\tensorflow\dataset_sample\pickle'
save_file = dataset_dir + '\dataset_{}.pkl'.format(name)

with open(save_file, 'rb') as f:
    dataset = pickle.load(f)
    x = dataset[0]
    t = dataset[1]
#==========================================================

import tensorflow as tf

holder_x = tf.placeholder(tf.float32)
holder_t = tf.placeholder(tf.float32)

w = tf.Variable(tf.random_normal([1],mean=0.0, stddev=1.0))
b = tf.Variable(tf.random_normal([1],mean=0.0, stddev=1.0))
y = holder_x * w + b

loss = tf.reduce_mean(tf.square(y - holder_t))

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(1001):
        if step % 100 == 0:
            print((step, sess.run(loss, feed_dict={holder_x: x,holder_t: t})))
        sess.run(train, feed_dict={holder_x: x,holder_t: t})

    weight = sess.run(w)
    bias = sess.run(b)

#Time required
e = time.time()
print('time:' + str(e-s))

#Plot graph
import matplotlib.pyplot as plt

plt.plot(x,t,'.')
plt.plot(x,x*weight+bias)
plt.ylim(0)
plt.show()
