'''
y = wx + b

입력값(x, y)은 placeholder
w, b -> variable
'''

import tensorflow as tf
from tensorflow.python.ops.random_ops import random_normal
sess = tf.Session()
tf.compat.v1.set_random_seed(66) # random state

x_train = [1, 2, 3]
y_train = [1, 2, 3]

x_train = tf.compat.v1.placeholder(tf.float32, shape=[None])
y_train = tf.compat.v1.placeholder(tf.float32, shape=[None])

# w = tf.Variable([1], dtype=tf.float32) # 랜덤하게 내가 넣어준 초기값 
# b = tf.Variable([1], dtype=tf.float32) # 초기값 

w = tf.Variable(tf.compat.v1.random_normal([1]), dtype=tf.float32) # random_normal -> 정규분포에 의한 random 
b = tf.Variable(tf.compat.v1.random_normal([1]), dtype=tf.float32) # random이지만, random_seed를 바꿔주지 않으면 계속 동일한 값으로 실행됨 

hypothesis = x_train*w+b # y = wx+b

loss = tf.reduce_mean(tf.square(hypothesis - y_train))
# mse


optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01) # Tensorflow에 존재하는 optimizer, learning_rate
train = optimizer.minimize(loss) # 1회전 했을 때 최솟값을 구해준다

sess.run(tf.compat.v1.global_variables_initializer())

for step in range(2001):
    # sess.run(train)
    _, loss_val, w_val, b_val = sess.run([train, loss, w, b], feed_dict={x_train:[1,2,3], y_train:[1,2,3]})
    if step % 20 == 0: # 100번 반복. 20번마다 출력한다 
        # print(step, sess.run(loss), sess.run(w), sess.run(b))
        # opt를 실행했을 때 갱신되는 loss, w, b를 출력한다 
        print(step, loss_val, w_val, b_val)






