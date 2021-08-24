import tensorflow as tf

print(tf.__version__)

print(tf.executing_eagerly()) # True

tf.compat.v1.disable_eager_execution()
# 즉시 실행 모드
# Tensorflow 2.x 에서도 1.x  형태로 테스트 할 수 있도록 도와준다 

print(tf.executing_eagerly()) # False

# print('Hello world')

# hello = tf.constant("Hello world")  

# print(hello)

# ssess = tf.compat.v1.Session()  
# print(ssess.run(hello))  
