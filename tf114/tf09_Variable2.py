import tensorflow as tf
tf.compat.v1.set_random_seed(777)

'''
실습 
방식 3가지로 출력 하시오 
'''


x = [1, 2, 3]
W = tf.Variable([0.3], tf.float32)
b = tf.Variable([1.0], tf.float32)

hypothesis = x * W + b


# 방법 1
sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())
aaa = sess.run(hypothesis)
print(aaa) 
sess.close()

# 방법 2
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
bbb =  hypothesis.eval() 
print(bbb)
sess.close()

# 방법 3
sess = tf.Session()
sess.run(tf.global_variables_initializer())
ccc = hypothesis.eval(session=sess)  
print(ccc)
sess.close()

# [1.3       1.6       1.9000001] 동일하게 출력


