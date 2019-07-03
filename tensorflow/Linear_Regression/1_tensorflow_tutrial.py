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

#I store dataset x and t in placeholder.
#And can calculate those values.
#In addition, I compare the values calculated in normal Python.

import tensorflow as tf

holder_x = tf.placeholder(tf.float32)
holder_t = tf.placeholder(tf.float32)
add_op = holder_x + holder_t

with tf.Session() as sess:
    result = sess.run(holder_x, feed_dict={holder_x: x})
    print(result)

    result = sess.run(holder_t, feed_dict={holder_t: t})
    print(result)

    result = sess.run(add_op, feed_dict={holder_x: x, holder_t: t})
    print(result)
  
print(x)
print(t)
print(x+t)


#Time required
e = time.time()
print('time:' + str(e-s))