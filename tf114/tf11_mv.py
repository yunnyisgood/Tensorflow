import tensorflow as tf
from tensorflow.python.ops.variables import global_variables_initializer
tf.set_random_seed(66)

'''
mult variable
x값이 여러개 일 때 
'''

x1_data  = [73., 93., 89., 96., 73.] # 국어 
x2_data  = [80., 88., 91., 98., 66.] # 영어 
x3_data  = [75., 93., 90., 100., 70.] # 수학  
y_data  = [152., 185., 180., 196., 142.] # 결과 

# x는 (5, 3), y는 (5, 1) 또는 (5, )

x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

w1 = tf.Variable(tf.random_normal([1]), name='weight1')
w2 = tf.Variable(tf.random_normal([1]), name='weight2')
w3 = tf.Variable(tf.random_normal([1]), name='weight3')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = x1 * w1 + x2*w2 + x3*w3 +b

cost = tf.reduce_mean(tf.square(hypothesis-y))

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01) => 러닝메이트 수치가 너무 커서 nan으로 값이 출력됨 
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.000001)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5) # 1e-5 = 0.00001
train = optimizer.minimize(cost)

sess = tf.compat.v1.Session()
sess.run(global_variables_initializer())

for epochs in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], 
                    feed_dict={x1:x1_data, x2:x2_data, x3:x3_data, y:y_data})
    if epochs % 10 == 0:
        print(epochs, 'cost: ', cost_val, "\n", hy_val)

sess.close()

