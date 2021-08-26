import numpy as np

# 함수형 모델의 layer명에 대한 설명

# 1. data
x = np.array([range(100), range(301, 401), range(1, 101),
             range(100), range(401, 501)])
x = np.transpose(x) # (100, 5)
print(x.shape)

y= np.array([range(711, 811), range(101, 201)])
y = np.transpose(y)
print(y.shape)

# 2. modeling
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

# 2) 함수형 모델
input1 = Input(shape=(5, ))
# hidden layer 부분-> 동일한 변수명 가능
# 단, 분기 없이  단일층에서만 사용가능
xx = Dense(3)(input1) 
xx = Dense(4)(xx)
xx = Dense(10)(xx)
output1 = Dense(2)(xx)

model = Model(inputs=input1, outputs=output1)

model.summary()

# 1) Sequential Model(순차적 모델)
# model = Sequential()
# model.add(Dense(3, input_shape = (5, )))
# model.add(Dense(4))
# model.add(Dense(10))
# model.add(Dense(2))

# model.summary()

# 3. compile
# model.compile()

# 4 evaluate