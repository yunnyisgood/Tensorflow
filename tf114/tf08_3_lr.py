'''
실습
tf08_2 파일의 lr을 수정해서
epoch가 2000번이 아닌 100번 이하로 줄여라
결과치는
step <= 100, w=1.99999, b=0.99999
'''

import tensorflow as tf
from tensorflow.python.ops.random_ops import random_normal
sess = tf.Session()
tf.compat.v1.set_random_seed(66) # random state
# tf.compat.v1.set_random_seed(42) # random state
# tf.compat.v1.set_random_seed(66) # random state
# tf.compat.v1.set_random_seed(9) # random state

x_train = tf.compat.v1.placeholder(tf.float32, shape=[None])
y_train = tf.compat.v1.placeholder(tf.float32, shape=[None])

w = tf.Variable(tf.compat.v1.random_normal([1]), dtype=tf.float32) # random_normal -> 정규분포에 의한 random 
b = tf.Variable(tf.compat.v1.random_normal([1]), dtype=tf.float32) # random이지만, random_seed를 바꿔주지 않으면 계속 동일한 값으로 실행됨 

hypothesis = x_train*w+b # y = wx+b

loss = tf.reduce_mean(tf.square(hypothesis - y_train))
# mse

optimizer = tf.train.AdamOptimizer(learning_rate=0.76) # Tensorflow에 존재하는 optimizer, learning_rate
train = optimizer.minimize(loss) # 1회전 했을 때 최솟값을 구해준다

sess.run(tf.compat.v1.global_variables_initializer())

for step in range(101):
    # sess.run(train)
    _, loss_val, w_val, b_val = sess.run([train, loss, w, b], feed_dict={x_train:[1,2,3], y_train:[3, 5, 7]}) # w=2, b=1
    if step % 20 == 0: # 100번 반복. 20번마다 출력한다 

        print(step, 'loss: ', loss_val, 'w: ',w_val, 'b: ', b_val)

# 100 loss:  9.856177e-05 w:  [1.9930507] b:  [0.9960421]