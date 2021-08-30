import tensorflow as tf
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers  import Conv2D
tf.set_random_seed(66)

# 1. data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.

# learning_rate = 0.001
learning_rate = 0.00001
training_epochs = 15
batch_size = 100
total_batch = int(len(x_train)/batch_size)

x = tf.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.placeholder(tf.float32, [None, 10])

# modeling
w1 = tf.get_variable('w1', shape=[3, 3, 1, 32]) #  자동으로 초기값을 넣어준다, 단 name, shape를 꼭 넣어줘야!
# shape=[3, 3, 1, 32]에서 3, 3은 커널사이즈 / 1은 x에서 받아들이는 채널의 수 /  32는  필터를 의미 

L1 = tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='VALID')
L3 = tf.nn.conv2d(x, w1, strides=[1, 2, 2, 1], padding='SAME')
L4 = tf.nn.conv2d(x, w1, strides=[1, 2, 2, 1], padding='VALID')

print(w1) # (3, 3, 1, 32)
print(L1) # (?, 28, 28, 32)
print(L2) # (?, 26, 26, 32) -> 28 -3 + 1 = 26 [stride가 1, kernerl이 3일 때]
print(L3) # (?, 14, 14, 32) -> 28개를 2의 간격으로 자르게 되면 14, 14모양으로
print(L4) # (?, 13, 13, 32) -> 14-3+2 = 13 [stride가 2, kernel이 3일때]


# max_pool test 하기
L1_maxpool = tf.nn.max_pool(L1, ksize=[1,2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
L2_maxpool = tf.nn.max_pool(L1, ksize=[1,1, 1, 1], strides=[1, 1, 1, 1], padding='SAME')


print(L1_maxpool)  # (?, 14, 14, 32)


# model = Sequential()
# model.add(Conv2D(filters=32, kernel_size=(3,3), strides=1,
#                     padding='same', input_shape=(28, 28, 1)))



'''
========get_variable과 Variable 차이=============
w2 = tf.Variable(tf.random_normal([3, 3, 1, 32], dtype=tf.float32))
w3 = tf.Variable([1], dtype=tf.float32)

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

print(np.min(sess.run(w1)))
print("====================")
print(np.max(sess.run(w1)))
print("====================")
print(np.mean(sess.run(w1)))
print("====================")
print(np.median(sess.run(w1)))
print("====================")
print(sess.run(w1))
print(w1)'''







