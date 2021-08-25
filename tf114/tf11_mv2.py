import tensorflow as tf
from tensorflow.python.ops.variables import global_variables_initializer
tf.set_random_seed(66)

x_data = [[73, 51, 65],   # (5, 3)
          [92, 98, 11],
          [89, 31, 33],
          [99, 33, 100],
          [17, 66, 79]]
y_data = [[152], [185], [100], [205], [142]]  # (5, 1)

x = tf.placeholder(tf.float32, shape=[None,3])  
y = tf.placeholder(tf.float32, shape=[None,1])  

w = tf.Variable(tf.random_normal([3, 1])) # 연산이 되기 위해서는 x값의 열과 w의 행이 같아야 한다 
b = tf.Variable(tf.random_normal([1]))

# hypothesis = x * w + b -> 행렬 연산이기에 적용될 수 없다 
hypothesis = tf.matmul(x, w) + b # 행렬 연산 

cost = tf.reduce_mean(tf.square(hypothesis-y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train = optimizer.minimize(cost)

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

for epochs in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], 
                    feed_dict={x:x_data, y:y_data})
    if epochs % 10 == 0:
        print(epochs, 'cost: ', cost_val, "\n", hy_val)

sess.close()

