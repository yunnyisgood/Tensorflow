import tensorflow as tf
from tensorflow.python.ops.variables import global_variables_initializer
tf.set_random_seed(66)

x_data = [[1,2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]  


x = tf.placeholder(tf.float32, shape=[None,2])  
y = tf.placeholder(tf.float32, shape=[None,1])  

w = tf.Variable(tf.random_normal([2, 1]))
b = tf.Variable(tf.random_normal([1]))
# x, b를 0에서 1사이의 랜덤한 값으로 지정

hypothesis = tf.sigmoid(tf.matmul(x, w) + b) # tf.sigmoid로 감싸면 0과 1사이의 값으로 나온다 

cost = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis)) # binary-crossentropy
# 원래의 y값과 hypotheseis된 y값을 비교 
# cost = tf.reduce_mean(tf.square(hypothesis-y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)



sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

for epochs in range(5001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], 
                    feed_dict={x:x_data, y:y_data})
    if epochs % 10 == 0:
        print(epochs, 'cost: ', cost_val, "\n", hy_val)

predict = tf.cast(hypothesis > 0.5, dtype=tf.float32)
# 0.5를 초과하면 1, 아니면 0
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, y), dtype=tf.float32))
c, a = sess.run([predict, accuracy], feed_dict={x:x_data, y:y_data})
print('predict:\n ', hy_val, "\n 예측결과 값\n : ", c, "\n Accuracy: ", a)


sess.close()

'''
predict:  [[0.07803829]
 [0.19524902]
 [0.4830764 ]
 [0.7082357 ]
 [0.8847676 ]
 [0.9640292 ]]
 원래 값 :  [[0.]
 [0.]
 [0.]
 [1.]
 [1.]
 [1.]] 
 Accuracy:  1.0
'''
