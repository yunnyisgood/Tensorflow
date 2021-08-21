import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. data
x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 2, 3, 4, 5])

# 2. model
model = Sequential()
model.add(Dense(3, input_dim=1)) # params 개수 : 3 X 1 + 3
model.add(Dense(2)) # params 개수: 3 X 2 + 2
model.add(Dense(1)) # params 개수: 2 X 1 + 1

model.summary()
'''
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 3)                 6
_________________________________________________________________
dense_1 (Dense)              (None, 2)                 8
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 3
=================================================================
Total params: 17
Trainable params: 17  -->  봐야 할 것 1
Non-trainable params: 0 -->  봐야 할 것 2
'''

print(model.weights) 
print('============================================')
print(model.trainable_weights)

# model.weights, model.trainable_weights 모두 동일하다 

print('============================================')
print(len(model.weights)) # 6 
print(len(model.trainable_weights)) # 6
# 6이 되는 이유는 층마다 weight, bias 가 1개씩. 층은 3개 => 3 X 2 = 6



'''
[<tf.Variable 'dense/kernel:0' <-- 첫번째 연산  shape=(1, 3) <- input이 1개, output 3개 , dtype=float32, 
numpy=array([[-0.9856683 , -0.28698182,  0.57491684]], dtype=float32)>, <= 3개의 연산 
<tf.Variable 'dense/bias:0' <-- bias 연산  shape=(3,) <-- bias 3 연산  dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>, 
<tf.Variable 'dense_1/kernel:0' shape=(3, 2) <-- 연산은 3 x 2 dtype=float32, numpy=

array([[ 0.16865528, -0.26033974],
       [ 0.62296903,  0.00137639],
       [-0.5100602 , -0.07348907]], dtype=float32)>, 
       <tf.Variable 'dense_1/bias:0' shape=(2,) dtype=float32, numpy=array([0., 0.], dtype=float32)>, 
       <tf.Variable 'dense_2/kernel:0' shape=(2, 1) dtype=float32, numpy=

array([[-0.6786846],
       [ 0.6970786]], dtype=float32)>, 
       <tf.Variable 'dense_2/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>]

'''





