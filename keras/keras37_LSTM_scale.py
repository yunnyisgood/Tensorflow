import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
from tensorflow.python.keras.layers.recurrent import LSTM

# data
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
             [5,6,7], [6,7,8],[7,8,9],[8,9,10],[9,10,11],[10,11,12],
             [20,30,40], [30,40,50], [40,50,60]]) 
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_pred = np.array([50,60,70])

x = x.reshape(x.shape[0], x.shape[1], 1)
x_pred = x_pred.reshape(1, x_pred.shape[0], 1)
# x_pred = x_pred.reshape(13, 3, 1)


print(x.shape, y.shape)  # (13, 3) (13,)


x = x.reshape(13,3,1) # (batch_size, timesteps, feature)
# 이 때 feature는 연산의 단위를 의미. 즉 몇개씩 자르는지 단위를 의미한다


# modeling
model = Sequential()
# model.add(SimpleRNN(units=10, activation='relu', input_shape=(3, 1)))
model.add(LSTM(units=32, activation='relu', input_shape=(3, 1)))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

model.summary()


# compile, fit
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=4)

# predict
y_pred = model.predict(x_pred)
print(y_pred)


'''
[[89.15195]]

[[82.61122]]

[[80.86775]]
'''

