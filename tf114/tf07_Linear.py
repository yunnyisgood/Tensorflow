'''
y = wx + b

입력값(x, y)은 placeholder
w, b -> variable
'''

import tensorflow as tf
sess = tf.Session()
tf.set_random_seed(66) # random state

x_train = [1, 2, 3]
y_train = [1, 2, 3]

w = tf.Variable([0000], dtype=tf.float32) # 랜덤하게 내가 넣어준 초기값 -> 아무값이나 넣어도 상관없다 
b = tf.Variable([1], dtype=tf.float32) # 초기값 

hypothesis = x_train*w+b # y = wx+b

loss = tf.reduce_mean(tf.square(hypothesis - y_train))
# mse


optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01) # Tensorflow에 존재하는 optimizer, learning_rate
train = optimizer.minimize(loss) # 1회전 했을 때 최솟값을 구해준다

sess.run(tf.global_variables_initializer())

for step in range(2001):
    sess.run(train)
    if step % 20 == 0: # 100번 반복. 20번마다 출력한다 
        print(step, sess.run(loss), sess.run(w), sess.run(b))
        # opt를 실행했을 때 갱신되는 loss, w, b를 출력한다 






