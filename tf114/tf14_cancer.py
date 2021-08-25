'''
accuracy_score
'''
from sklearn.datasets import load_breast_cancer
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

tf.set_random_seed(66)

datasets = load_breast_cancer()
x_data = datasets.data
y_data = datasets.target
y_data = y_data.reshape(569, 1)
print(x_data.shape, y_data.shape)
# (569, 30) (569,)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, 
train_size=0.8, shuffle=True, random_state=66)

x = tf.placeholder(tf.float32, shape=[None, 30])
y = tf.placeholder(tf.float32, shape=[None, 1])


w = tf.Variable(tf.random_normal([30, 1]), dtype=tf.float32)
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32)

hypothesis = tf.sigmoid(tf.matmul(x, w) + b)

cost = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis)) 

optimizer = tf.train.GradientDescentOptimizer(learning_rate=9e-1)
train = optimizer.minimize(cost)



sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

for epochs in range(5001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], 
                    feed_dict={x:x_train, y:y_train})
    if epochs % 10 == 0:
        print(epochs, 'cost: ', cost_val, "\n", hy_val)

predict = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, y), dtype=tf.float32))

h, c, a = sess.run([hypothesis ,predict, accuracy], feed_dict={x:x_test, y:y_test})
print('predict:\n ', hy_val, "\n 예측결과 값\n : ", c, "\n Accuracy: ", a)



sess.close()
