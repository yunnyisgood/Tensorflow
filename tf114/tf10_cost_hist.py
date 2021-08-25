import tensorflow as tf
import matplotlib.pyplot as plt

x = [1., 2., 3.]
y = [2., 4., 6.]

W = tf.placeholder(tf.float32)

hypothesis = x * W  # bias 없는 데이터

cost = tf.reduce_mean(tf.square(hypothesis-y)) # loss를 cost라고도 한다

w_history = []
cost_history = [] # loss history

with tf.compat.v1.Session() as sess:
    for i in range(-30, 50):
        curr_w = i
        curr_cost = sess.run(cost, feed_dict={W:curr_w})
        
        w_history.append(curr_w)
        cost_history.append(curr_cost)

print("===============w_history=-================")
print(w_history)
print("===============cost_history=-================")
print(cost_history)


plt.plot(w_history, cost_history)
plt.xlabel('weight')
plt.ylabel('loss')
plt.show()