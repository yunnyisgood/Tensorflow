from sklearn.datasets import load_diabetes
import tensorflow as tf
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

tf.set_random_seed(66)

datasets = load_diabetes()
x_data = datasets.data
y_data = datasets.target
y_data = y_data.reshape(442, 1)
print(x_data.shape, y_data.shape)
# (442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, 
train_size=0.8, shuffle=True, random_state=66)

x = tf.placeholder(tf.float32, shape=[None, 10])
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random_normal([10, 1]), dtype=tf.float32)
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32)

hypothesis = tf.matmul(x, w) + b

cost = tf.reduce_mean(tf.square(hypothesis-y))

optimizer = tf.train.AdamOptimizer(learning_rate=5e-1)
train = optimizer.minimize(cost)

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

for epochs in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], 
                    feed_dict={x:x_train, y:y_train})
    if epochs % 10 == 0:
        print(epochs, 'cost: ', cost_val, "predict: ", hy_val)

pred = sess.run(hypothesis, feed_dict={x:x_test})
score = r2_score(y_test, pred)
print('r2_score: ', score)
sess.close()

# r2_score:  0.499216339407228