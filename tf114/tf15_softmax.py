import numpy as np
import tensorflow as tf

tf.set_random_seed(66)

x_data =[[1, 2, 1, 1],
        [2, 1, 3, 2],
        [3, 1, 3, 4],
        [4, 1, 5, 5],
        [1, 7, 5, 5],
        [1, 2, 5, 6],
        [1, 6, 6, 6],
        [1, 7, 6, 7]] # (8, 4)

y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]] # (8, 3)


x = tf.placeholder(tf.float32, shape=[None, 4])
y = tf.placeholder(tf.float32, shape=[None, 3])

w = tf.Variable(tf.random_normal([4, 3]), dtype=tf.float32) # x X w = N, 3 (y)의 값이 되어야 한다 
b = tf.Variable(tf.random_normal([1, 3]), dtype=tf.float32) 
# 처음에 x에 넣어줘야 하기 때문에 [1, ] 그리고 출력은 y 형태와 같아야 하기 때문에 [, 3]의 형태 => 총 [1, 3]

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


results = sess.run(hypothesis,feed_dict={x:[[1, 11, 7, 9]]})
print('예측결과 : ',results)
# print(results, sess.run(tf.argmax(results, 1)))

