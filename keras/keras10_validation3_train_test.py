from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 1. 데이터
# 훈련 데이터 
x = np.array([1, 2, 4, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13])
y = np.array([1, 2, 4, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13])

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6, 
random_state=66, shuffle=True)
# train과 test를 6:4로 나눈다음

x_test, x_val, y_test, y_val = train_test_split(x, y, test_size=0.5, random_state=66,
shuffle=True)
# test와  val을 다시 5:5로 나눈다. 

# 다시 정리해서 결과 도출해 보기

print(x_train.shape, y_train.shape)  
print(x_test.shape, y_test.shape) 

# 2. model

model =Sequential()

model.add(Dense(10, input_dim=1))
model.add(Dense(3))
model.add(Dense(7))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(1))

# compile
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

model.fit()
