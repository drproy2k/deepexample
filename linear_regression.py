##################################################################
# linear_regression.py
# 간단 예제
#

# 1. 일변수 예제
import tensorflow as tf
#import matplotlib.pyplot as plt

# 신경망에서는 가중치를 '0'이 아니라 난수로 초기화해야만 학습이 가능합니다.
W1 = tf.Variable(tf.random_uniform([1, 10], -1, 1))
b1 = tf.Variable(tf.zeros([10]))
W2 = tf.Variable(tf.random_uniform([10, 1], -1, 1))
b2 = tf.Variable(tf.zeros([1]))

x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

x_data = [[0.0],
          [0.1],
          [0.2],
          [0.3],
          [0.4],
          [0.5],
          [0.6],
          [0.8]]
y_data = [[.02],
          [.03],
          [.06],
          [.11],
          [.18],
          [.27],
          [.38],
          [.66]]

L1 = tf.nn.relu(tf.matmul(x, W1) + b1)

model = tf.matmul(L1, W2) + b2
loss = tf.reduce_sum(tf.square(model - y))

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(10000):
    sess.run(train, feed_dict={x: x_data, y: y_data})

print("x=7", "y=?",  sess.run(model, feed_dict={x: [[.7]]}))

print("------------------------------------")
print("W1", sess.run(W1))
print("W2", sess.run(W2))
print("b1", sess.run(b1))
print("b2", sess.run(b2))
print("loss", sess.run(loss, feed_dict={x: x_data, y: y_data}))

# %matplotlib inline
# y_ = sess.run(model, feed_dict={x: x_data})
# plt.plot(x_data, y_data, 'ro')
# plt.plot(x_data, y_)
# plt.show()

input('paused.. 1')

###################################################
# 2. 다변수 예제
import tensorflow as tf
import numpy as np

W = tf.Variable(tf.zeros([3, 1]))
b = tf.Variable(tf.zeros([1]))

x = tf.placeholder(tf.float32, [None, 3])
y = tf.placeholder(tf.float32, [None, 1])

x_data = [[0, 0, 0],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 1],
          [0, 2, 0],
          [0, 2, 1],
          [1, 0, 0],
          [1, 0, 1],
          [1, 1, 0],
          [1, 1, 1],
          [1, 2, 0],
          [1, 2, 1]]
'''
y_data = [[3], 
          [4], 
          [6], 
          [7], 
          [9], 
          [10], 
          [5], 
          [6], 
          [8], 
          [9], 
          [11], 
          [12]]
'''
y_data_raw = np.array([3, 4, 6, 7, 9, 10, 5, 6, 8, 9, 11, 12])
y_data = y_data_raw.reshape(-1, 1)      # 위의 주석처리 된 상태처럼 된다

linear_model = tf.matmul(x, W) + b
loss = tf.reduce_sum(tf.square(linear_model - y))

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
    sess.run(train, feed_dict={x: x_data, y: y_data})

print("W", sess.run(W, feed_dict={x: x_data, y: y_data}))
print("b", sess.run(b, feed_dict={x: x_data, y: y_data}))
print("loss", sess.run(loss, feed_dict={x: x_data, y: y_data}))

print("y", sess.run(linear_model, feed_dict={x: [[2, 3, 4], [5, 6, 7]]}))

input('pause.. 2')

#################################################
# 3. 다층구조 예제
import tensorflow as tf
#import matplotlib.pyplot as plt

W1 = tf.Variable(tf.random_uniform([1, 10], -1, 1))
b1 = tf.Variable(tf.zeros([10]))
W2 = tf.Variable(tf.random_uniform([10, 10], -1, 1))
b2 = tf.Variable(tf.zeros([10]))
W3 = tf.Variable(tf.random_uniform([10, 1], -1, 1))
b3 = tf.Variable(tf.zeros([1]))

x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

x_data = [[0.0],
          [0.1],
          [0.2],
          [0.3],
          [0.4],
          [0.5],
          [0.6],
          [0.8]]
y_data = [[.02],
          [.03],
          [.06],
          [.11],
          [.18],
          [.27],
          [.38],
          [.66]]

L1 = tf.nn.relu(tf.matmul(x, W1) + b1)
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)

model = tf.matmul(L2, W3) + b3
loss = tf.reduce_sum(tf.square(model - y))

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(10000):
    sess.run(train, feed_dict={x: x_data, y: y_data})

print("x=7", "y=?",  sess.run(model, feed_dict={x: [[.7]]}))

print("------------------------------------")
print("W1", sess.run(W1))
print("W2", sess.run(W2))
print("W3", sess.run(W3))
print("b1", sess.run(b1))
print("b2", sess.run(b2))
print("b3", sess.run(b3))
print("loss", sess.run(loss, feed_dict={x: x_data, y: y_data}))

#%matplotlib inline
y_ = sess.run(model, feed_dict={x: x_data})
# plt.plot(x_data, y_data, 'ro')
# plt.plot(x_data, y_)
# plt.show()

# import matplotlib
# import numpy as np
#
# %matplotlib inline
# plt.imshow(x_data)
# plt.show()
#
# %matplotlib inline
# plt.imshow(y_data)
# plt.show()
#
# %matplotlib inline
# plt.imshow(sess.run(W1))
# plt.show()
#
# %matplotlib inline
# plt.imshow(sess.run(W2))
# plt.show()
#
# %matplotlib inline
# plt.imshow(sess.run(W3))
# plt.show()