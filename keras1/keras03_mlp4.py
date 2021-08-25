import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt


# 1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,4,3,5,7,9,3,8,12])

z = np.array([[[1,2],[3,4],[5,6]]])
print(z.shape)

# 2. model
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(4))
model.add(Dense(5))
model.add(Dense(7))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x, y, epochs=100, batch_size=1)

#4. 평가 예측
loss = model.evaluate(x, y)
print('loss: ', loss)

# loss:  3.6715245246887207

result = model.predict([10])
print('result: ', result)

# result:  [[9.510752]]

y_pred = model.predict(x)
plt.scatter(x, y)
plt.plot(x, y_pred, color='red')
plt.show()

