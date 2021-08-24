import tensorflow as tf
sess = tf.Session()

x = tf.Variable([2], dtype=tf.float32, name='test')

init = tf.global_variables_initializer()
# 변수 초기화 -> Tensorflow 형태에 적합한 자료형의 구조로 초기화 시킨다 

sess.run(init)
print(sess.run(x))

