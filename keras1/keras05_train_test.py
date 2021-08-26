from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

# 1. 데이터
# 훈련 데이터 
x_train = np.array([1, 2, 4, 3, 5, 6, 7])
y_train = np.array([1, 2, 4, 3, 5, 6, 7])

# 평가 데이터
x_test = np.array([8,9,10])
y_test = np.array([8,9,10])

# 2. model
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(4))
model.add(Dense(5))
model.add(Dense(7))
model.add(Dense(3))
model.add(Dense(1))


# 3. Compile
model.compile(loss='mse', optimizer="adam")

model.fit(x_train, y_train, epochs=1000, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)

# loss:  3.2621681690216064

# result = model.predict([x_pred])
# print('10의 예측값: ', result)

# 10의 예측값:  [[9.276111]]

y_pred = model.predict([11])
# plt.scatter(x, y)
# plt.plot(x, y_pred, color='red')
# plt.show()