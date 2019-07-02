import time
s = time.time()

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf


const1 = tf.constant(2)
const2 = tf.constant(3)
add_op = const1 + const2
mul_op = const1 * const2

with tf.Session() as sess:
  result, result2 = sess.run([mul_op, add_op])
  print(result)
  print(result2)


e = time.time()
print('time:' + str(e-s))