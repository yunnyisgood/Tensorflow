import tensorflow as tf

print(tf.__version__)

print('Hello world')

hello = tf.constant("Hello world") # 문자를 상수로 정의한다 

print(hello)
# Tensor("Const:0", shape=(), dtype=string) -> 그냥 print 하면 자료구조가 출력됨 

# sess = tf.Session() #  session을 정의하고 
sess = tf.compat.v1.Session()
print(sess.run(hello)) # run 해줘야 한다 
# b'Hello world'

# -> hello라는 상수를 넣고 sess.run()을 해줘야 문자열 출력