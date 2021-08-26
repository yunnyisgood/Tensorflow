from keras.datasets import mnist
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np
from sklearn.metrics import accuracy_score
tf.set_random_seed(66)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)  # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)  # (10000, 28, 28) (10000,)

x_train = x_train.reshape(60000, 28 *28)/255.
y_train = y_train.reshape(60000, 1)
x_test = x_test.reshape(10000, 28 *28)/255.
y_test = y_test.reshape(10000, 1)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


one_hot_Encoder = OneHotEncoder()
one_hot_Encoder.fit(y_train)
y_train = one_hot_Encoder.transform(y_train).toarray()
y_test = one_hot_Encoder.transform(y_test).toarray()
# y_train = tf.keras.utils.to_categorical(y_train)
# y_test = tf.keras.utils.to_categorical(y_test)

print(x_train.shape, y_train.shape)  # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)  # (10000, 28, 28) (10000,)

# 2. modeling
x = tf.placeholder(tf.float32, shape=[None,28*28])  
y = tf.placeholder(tf.float32, shape=[None,10])  

# hidden layer 1
w1 = tf.Variable(tf.random.normal([28*28, 5000],stddev=0.1))
b1 = tf.Variable(tf.random.normal([1, 5000],stddev=0.1))
# x, b를 0에서 1사이의 랜덤한 값으로 지정

layer1 = tf.nn.relu(tf.matmul(x, w1) + b1) 
lyaer_d = tf.nn.dropout(layer1, keep_prob=0.25)

# hidden layer 2
w2 = tf.Variable(tf.random.normal([5000, 500],stddev=0.1))
b2 = tf.Variable(tf.random.normal([1, 500],stddev=0.1))
# x, b를 0에서 1사이의 랜덤한 값으로 지정

layer2 = tf.nn.relu(tf.matmul(lyaer_d, w2) + b2)

# hidden layer 3
w3 = tf.Variable(tf.random.normal([500, 64],stddev=0.1))
b3 = tf.Variable(tf.random.normal([1, 64],stddev=0.1))
# x, b를 0에서 1사이의 랜덤한 값으로 지정

layer3 = tf.nn.relu(tf.matmul(layer2, w3) + b3)

# hidden layer 4
w4 = tf.Variable(tf.random.normal([64, 20],stddev=0.1))
b4 = tf.Variable(tf.random.normal([1, 20],stddev=0.1))
# x, b를 0에서 1사이의 랜덤한 값으로 지정

layer4 = tf.nn.relu(tf.matmul(layer3, w4) + b4)

# output layer
w = tf.Variable(tf.random.normal([20, 10],stddev=0.1))
b = tf.Variable(tf.random.normal([1, 10],stddev=0.1))

hypothesis = tf.nn.softmax(tf.matmul(layer4, w) + b) 

cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))

optimizer = tf.train.AdamOptimizer(learning_rate=0.0011).minimize(cost)

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

for epochs in range(101):
    cost_val, hy_val = sess.run([cost, hypothesis], 
                    feed_dict={x:x_train, y:y_train})
    if epochs % 10 == 0:
        print(epochs, 'cost: ', cost_val, "\n", hy_val)

pred = sess.run(hypothesis, feed_dict={x:x_test}) # x_test에 대한 기울기 다시 도출 
print('전:', pred[:5])
pred = np.argmax(pred, axis=1)
print('후:', pred[:5])
print('전 1:', y_test[:5])
y_test = np.argmax(y_test, axis=1)
print('후 1:', y_test[:5])
print('acc: ', accuracy_score(y_test, pred))

# pred = sess.run(hypothesis, feed_dict={x:x_test}) # x_test에 대한 기울기 다시 도출 
# print('acc: ', accuracy_score(y_test, pred))
