import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt


# 1. 데이터
x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
             [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],
             [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]])
print(x.shape)
x = np.transpose(x)
# x2 = x.swapexse(0, 1)
print(x.shape)  # (10, 3)

y = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20]) # (10,)
print(y.shape) # (10, )


# 2. model
model = Sequential()
# model.add(Dense(5, input_dim=3))
# 차원이 2차원을 넘어선다면 input_dim으로 표현할 수 X
# -> input_shape 사용해야 함
model.add(Dense(5, input_shpae=(3,))) 
# input_shape = (3, ) -> 특성 3개, column 3개를 지정하겠다는 의미
# input_shape -> 행을 무시, 열 우선 
# (100, 4, 5, 3) ㅡ> data=100개, 실제 input_shpae=(4, 5, 3)
# 특성만 맞는다면 데이터의 개수와 상관없이 modeling 할 수 있다.
# 실제 데이터가 (10, 3)이라면 input_shape = (3, )이 된다.


model.add(Dense(4))
model.add(Dense(5))
model.add(Dense(7))
model.add(Dense(2))
model.add(Dense(1))

# 3. compile
model.compile(loss="mse", optimizer="adam")

model.fit(x, y, epochs=1000, batch_size=1)

#4. 평가 예측
loss = model.evaluate(x, y)
print('loss: ', loss)

# loss:  3.352187923155725e-05

result = model.predict(x_pred)
print('result: ', result)

# result:  [[  3.7012   28.30966 208.91016]]

y_pred = model.predict(x)
plt.scatter(x[:,0], y)
plt.scatter(x[:,1], y)
plt.scatter(x[:,2], y)
plt.plot(x, y_pred, color='red')
plt.show()