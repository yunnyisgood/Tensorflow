import tensorflow as tf
from tensorflow.python.ops.variables import global_variables_initializer
tf.compat.v1.set_random_seed(777)

# 변수를 사용하는 여러가지 방법 

# 방법 1
W = tf.Variable(tf.random_normal([1], name='weight'))
print(W)
# <tf.Variable 'Variable:0' shape=(1,) dtype=float32_ref>

sess = tf.compat.v1.Session()
sess.run(global_variables_initializer())
aaa = sess.run(W)
print(aaa) 
sess.close()

# 방법 2
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
bbb =  W.eval() 
print(bbb)
sess.close()

# 방법 3
sess = tf.Session()
sess.run(tf.global_variables_initializer())
ccc = W.eval(session=sess) 
print(ccc)
sess.close()

# [2.2086694] 동일하게 출력됨 