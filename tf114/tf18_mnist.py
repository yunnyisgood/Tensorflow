from keras.datasets import mnist
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np
from sklearn.metrics import accuracy_score
tf.set_random_seed(66)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)  # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)  # (10000, 28, 28) (10000,)


x_train = x_train.reshape(60000, 28 *28)
y_train = y_train.reshape(60000, 1)
x_test = x_test.reshape(10000, 28 *28)
y_test = y_test.reshape(10000, 1)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

print(x_train.shape, y_train.shape)  # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)  # (10000, 28, 28) (10000,)
# 2. modeling
x = tf.placeholder(tf.float32, shape=[None,28*28])  
y = tf.placeholder(tf.float32, shape=[None,10])  

# hidden layer 1
w = tf.Variable(tf.random_normal([28*28, 10]))
b = tf.Variable(tf.random_normal([10]))

hypothesis = tf.nn.softmax(tf.matmul(x, w) + b) 

cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))

optimizer = tf.train.AdamOptimizer(learning_rate=0.000011).minimize(cost)

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

for epochs in range(1001):
    cost_val, hy_val = sess.run([cost, hypothesis], 
                    feed_dict={x:x_train, y:y_train})
    if epochs % 10 == 0:
        print(epochs, 'cost: ', cost_val, "\n", hy_val)

'''pred = sess.run(hypothesis, feed_dict={x:x_test}) # x_test에 대한 기울기 다시 도출 
print('전:', pred[:5])
pred = np.argmax(pred, axis=1)
print('후:', pred[:5])
print('전 1:', y_test[:5])
y_test = np.argmax(y_test, axis=1)
print('후 1:', y_test[:5])
print('acc: ', accuracy_score(y_test, pred))'''

predict = tf.cast(hypothesis > 0.5, dtype=tf.float32)
# 0.5를 초과하면 1, 아니면 0
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, y), dtype=tf.float32))
c, a = sess.run([predict, accuracy], feed_dict={x:x_test, y:y_test})
print('predict:\n ', hy_val, "\n 예측결과 값\n : ", c, "\n Accuracy: ", a)