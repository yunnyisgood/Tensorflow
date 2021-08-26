
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score
import numpy as np

datasets = load_wine()
print(datasets.DESCR)
print(datasets.feature_names)

x_data = datasets.data
y_data = datasets.target
y_data = y_data.reshape(-1, 1)
print(x_data.shape, y_data.shape)  # (178, 13) (178, 1)

one_hot_Encoder = OneHotEncoder()
one_hot_Encoder.fit(y_data)
y_data = one_hot_Encoder.transform(y_data).toarray()

print(x_data.shape, y_data.shape)  # (178, 13) (178, 3)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.8, random_state=9, shuffle=True)

x = tf.placeholder(tf.float32, shape=[None, 13])
y = tf.placeholder(tf.float32, shape=[None, 3])

w = tf.Variable(tf.random_normal([13, 3]), dtype=tf.float32) # x X w = N, 3 (y)의 값이 되어야 한다 
b = tf.Variable(tf.random_normal([1, 3]), dtype=tf.float32) 

hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)

loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1)) # categorical crossentropy

optimizer = tf.train.GradientDescentOptimizer(learning_rate=5e-1).minimize(loss)

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

# acc:  0.4722222222222222