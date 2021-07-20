import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
from tensorflow.python.ops.gen_array_ops import Gather

# data
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]]) # 하나의 묶음단위를 timestpes라고 한다 
y = np.array([4,5,6,7])

print(x.shape, y.shape) 
# (4, 3) (4,)
# 몇개씩 잘라서 연산을 할지 결정해줘야 한다

x = x.reshape(4, 3, 1) # (batch_size, timesteps, feature)
# 이 때 feature는 연산의 단위를 의미. 즉 몇개씩 자르는지 단위를 의미한다


# modeling
# modeling
model = Sequential()
# model.add(SimpleRNN(units=10, activation='relu', input_shape=(3, 1)))
model.add(GRU(units=32, activation='relu', input_shape=(3, 1)))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

model.summary()


# compile, fit
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)

# predict
x_input = np.array([5,6,7]).reshape(1,3,1)
y_pred = model.predict(x_input)
print(y_pred)


'''
[[8.985677]]

[[8.3930025]]

[[8.332887]]

[[8.198724]]

'''
