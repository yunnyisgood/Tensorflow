
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import numpy as np

datasets = load_iris()
print(datasets.DESCR)
print(datasets.feature_names)

x_data = datasets.data
y_data = datasets.target
y_data = y_data.reshape(-1, 1)
print(x_data.shape, y_data.shape)  # (150, 4) (150,)

# y_data = tf.one_hot(y_data, 3).to_numpy()
# print(y_data.shape) # (150, 3)

one_hot_Encoder = OneHotEncoder()
one_hot_Encoder.fit(y_data)
y_data = one_hot_Encoder.transform(y_data).toarray()

print(type(x_data), type(y_data)) # <class 'numpy.ndarray'> <class 'numpy.ndarray'>

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.8, random_state=9, shuffle=True)

x = tf.placeholder(tf.float32, shape=[None, 4])
y = tf.placeholder(tf.float32, shape=[None, 3])

w = tf.Variable(tf.random_normal([4, 3]), dtype=tf.float32) # x X w = N, 3 (y)의 값이 되어야 한다 
b = tf.Variable(tf.random_normal([1, 3]), dtype=tf.float32) 

hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)

loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1)) # categorical crossentropy

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())                                                       

for epochs in range(1001):
    cost_val, hy_val, _ = sess.run([loss, hypothesis, optimizer], 
                    feed_dict={x:x_data, y:y_data})
    if epochs % 10 == 0:
        print(epochs, 'cost: ', cost_val, "predict: ", hy_val)

pred = sess.run(hypothesis, feed_dict={x:x_test}) # x_test에 대한 기울기 다시 도출 

print('전:', pred[:5])
pred = np.argmax(pred, axis=1)
print('후:', pred[:5])

print('전 1:', y_test[:5])
y_test = np.argmax(y_test, axis=1)
print('후 1:', y_test[:5])

print('acc: ', accuracy_score(y_test, pred))



'''
전: 
[[0.00425841 0.45491272 0.54082894]
 [0.03159882 0.4823888  0.48601234]
 [0.00967443 0.35125422 0.63907135]
 [0.01274231 0.35113665 0.63612103]
 [0.03131288 0.70819366 0.26049346]]
후: [2 2 2 2 1]

전 1: 
[[0. 0. 1.]
 [0. 1. 0.]
 [0. 0. 1.]
 [0. 0. 1.]
 [0. 1. 0.]]
후 1: [2 1 2 2 1]
'''


