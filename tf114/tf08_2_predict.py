'''
실습
1. [4]
2. [5, 6]
1. [6, 7, 8]

'''

import tensorflow as tf
from tensorflow.python.ops.random_ops import random_normal
sess = tf.Session()
tf.compat.v1.set_random_seed(66) # random state

x_train = [1, 2, 3]
y_train = [1, 2, 3]

x_train = tf.compat.v1.placeholder(tf.float32, shape=[None])
y_train = tf.compat.v1.placeholder(tf.float32, shape=[None])

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
    _, loss_val, w_val, b_val = sess.run([train, loss, w, b], feed_dict={x_train:[1,2,3], y_train:[3, 5, 7]}) # w=2, b=1
    if step % 20 == 0: # 100번 반복. 20번마다 출력한다 
        # print(step, sess.run(loss), sess.run(w), sess.run(b))
        # opt를 실행했을 때 갱신되는 loss, w, b를 출력한다 
        print(step, loss_val, w_val, b_val)

# predict 하기
# x_test라는 placeholder 만들기

x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])

hypothesis_test = x_test *w_val + b_val # 이미 학습을 통한 결과값 w_val, b_val을 사용하여 기울기 정의 

print('[4] predict: ', sess.run(hypothesis_test, feed_dict={x_test:[4]}))
print('[5, 6] predict: ', sess.run(hypothesis_test, feed_dict={x_test:[5, 6]}))
print('[6, 7, 8] predict: ', sess.run(hypothesis_test, feed_dict={x_test:[6, 7, 8 ]}))


'''
[4] predict:  [8.998177]
[5, 6] predict:  [10.997122 12.996066]
[6, 7, 8] predict:  [12.996066 14.99501  16.993954]
'''