import tensorflow as tf
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers  import Conv2D
from tensorflow.python.ops.gen_data_flow_ops import PaddingFIFOQueue
tf.set_random_seed(66)

# 1. data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.

learning_rate = 0.001
training_epochs = 1
batch_size = 64
total_batch = int(len(x_train)/batch_size)

x = tf.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.placeholder(tf.float32, [None, 10])

# modeling
w1 = tf.get_variable('w1', shape=[3, 3, 1, 32]) #  자동으로 초기값을 넣어준다, 단 name, shape를 꼭 넣어줘야!
# shape=[3, 3, 1, 32]에서 3, 3은 커널사이즈 / 1은 x에서 받아들이는 채널의 수 /  32는  필터를 의미 

# Layer 1
L1 = tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1) # layer에 activation 적용
L1_maxpool = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') # layer에 activation 적용
# L1_maxpool = tf.nn.max_pool(L1, ksize=2, strides=2, padding='SAME') # layer에 activation 적용
# ksize, strides 모두 (2,2)가 default가 되도록 처리 -> 위의 두개 모두 동일한 의미 

# model = Sequential()
# model.add(Conv2D(filters=32, kernel_size=(3,3), strides=1,
#                     padding='same', input_shape=(28, 28, 1),
#                       activation='relu'))

print(w1) # (3, 3, 1, 32)
print(L1) # (?, 28, 28, 32)
print(L1_maxpool) # (?, 14, 14, 32)

# Layer 2
w2 = tf.get_variable('w2', shape=[3, 3, 32, 64]) # 이 때 32는 그 전 layer에서 output의 수가 되어야
L2 = tf.nn.conv2d(L1_maxpool, w2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.selu(L2)
L2_maxpool = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# padding, strides 에 따라 어떻게 shape이 바뀌는지 다시 

print(L2) # (?, 14, 14, 64)
print(L2_maxpool) # (?, 7, 7, 64)


# Layer 3
w3 = tf.get_variable('w3', shape=[3, 3, 64, 128]) # 이 때 32는 그 전 layer에서 output의 수가 되어야
L3 = tf.nn.conv2d(L2_maxpool, w3, strides=[1, 1, 1, 1], padding='SAME')
L3 = tf.nn.elu(L3)
L3_maxpool = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# 

print(L3) # (?, 7, 7, 128)
print(L3_maxpool) # (?, 4, 4, 128)


# Layer 4
w4 = tf.get_variable('w4', shape=[2, 2, 128, 64], 
                    initializer=tf.contrib.layers.xavier_initializer()) # 이 때 32는 그 전 layer에서 output의 수가 되어야
L4 = tf.nn.conv2d(L3_maxpool, w4, strides=[1, 1, 1, 1], padding='VALID')
# valid 하게 되면 그 전 layer의 커널 사이즈를 다시 2, 2 의 모양으로 잘라서 사용할 것을 의미 
L4 = tf.nn.leaky_relu(L4)
L4_maxpool = tf.nn.max_pool(L4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# 

print(L4) # (?, 3, 3, 64)
print(L4_maxpool) # (?, 2, 2, 64) 

# Flatten
L_flat = tf.reshape(L4_maxpool, [-1, 2*2*64])
print('Flatten: ', L_flat) # (?, 256)

# L5
w5 = tf.get_variable('w5', shape=[256, 64],
                        initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([64]))
L5 = tf.nn.selu(tf.matmul(L_flat, w5)+ b5)
L5 = tf.nn.dropout(L5, keep_prob=0.2)
print(L5)  # (?, 64)

# L6
w6 = tf.get_variable('w6', shape=[64, 32],
                        initializer=tf.contrib.layers.xavier_initializer())
b6 = tf.Variable(tf.random_normal([32]))
L6 = tf.nn.selu(tf.matmul(L5, w6)+ b6)
L6 = tf.nn.dropout(L6, keep_prob=0.2)
print(L6)  # shape=(?, 32)

# L7
w7 = tf.get_variable('w7', shape=[32, 10])
b7 = tf.Variable(tf.random_normal([10]))
hypothesis = tf.nn.softmax(tf.matmul(L6, w7)+ b7)
print(hypothesis) # shape=(?, 10)


loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(training_epochs):
    avg_loss = 0

    for i in range(total_batch): # 1 epoch당 600번 훈련
        start = i * batch_size
        end = start + batch_size
        batch_x, batch_y = x_train[start:end], y_train[start:end]

        feed_dict = {x: batch_x, y:batch_y}

        batch_loss, _ = sess.run([loss, optimizer], feed_dict=feed_dict)

        avg_loss = batch_loss/total_batch
    # print(i, 'avg_loss: ', avg_loss)
    print('Epoch: ', '%04d' %(epoch+1), 'loss: {:.9f}'.format(avg_loss))

pred = tf.equal(tf.compat.v1.arg_max(hypothesis, 1), tf.compat.v1.argmax(y,1))
acc = tf.reduce_mean(tf.cast(pred, tf.float32))
print('acc: ', sess.run(acc, feed_dict={x:x_test, y:y_test}))



'''
acc 0.996 이상

batch_size = 64



batch_size = 128
0.5178

batch_size = 256
acc:  0.4015

batch_size = 512
acc:  0.2903
'''