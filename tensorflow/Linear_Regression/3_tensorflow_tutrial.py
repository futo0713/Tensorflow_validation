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

name = 'dataset_linear'
dataset_dir = 'C:\\Users\FutoshiTsutsumi\Desktop\python_test\\tensorflow\dataset_sample\pickle'
save_file = dataset_dir + '\dataset_linear.pkl'

with open(save_file, 'rb') as f:
    dataset = pickle.load(f)
    x = dataset[0]
    t = dataset[1]
#==========================================================

#I store x on placeholder. 
#And I set the variable; w and b.
#I calculate Affine.

import tensorflow as tf

holder_x = tf.placeholder(tf.float32)
holder_t = tf.placeholder(tf.float32)

w = tf.Variable(tf.random_normal([1],mean=0.0, stddev=1.0))
b = tf.Variable(tf.random_normal([1],mean=0.0, stddev=1.0))

affine = holder_x * w + b

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    result_w = sess.run(w)
    print(result_w)

    result_b = sess.run(b)
    print(result_b)

    result_x = sess.run(holder_x, feed_dict={holder_x: x})
    print(result_x)

    result = sess.run(affine, feed_dict={holder_x: x})
    print(result)


#Time required
e = time.time()
print('time:' + str(e-s))