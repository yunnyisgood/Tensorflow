import numpy as np

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
dense1 = Dense(3)(input1) # 상위 레이어를 뒤에 명시해준다 
dense2 = Dense(4)(dense1)
dense3 = Dense(10)(dense2)
output1 = Dense(2)(dense3)

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