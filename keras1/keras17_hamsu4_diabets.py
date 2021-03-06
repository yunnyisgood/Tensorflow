import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

# 당뇨 예측

# 1. 데이터
datasets  = load_diabetes()
x = datasets.data
y = datasets.target

print(x.shape) # (442, 10)
print(y.shape) # (442, )

print(datasets.feature_names)
# 'age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
# 10가지의 지수를 통해 당뇨병 지수를 파악한다 

print(datasets.DESCR)

x_train, x_test, y_train, y_test = train_test_split(x, y, 
train_size=0.7, shuffle=True, random_state=9)

# 2. modeling

input1 = Input(shape=(10, ))
dense1 = Dense(1000)(input1)
dense2 = Dense(60)(dense1)
dense3 = Dense(40)(dense2)
dense4 = Dense(40)(dense3)
dense5 = Dense(24)(dense4)
dense6 = Dense(24)(dense5)
dense7 = Dense(15)(dense6)
dense8 = Dense(8)(dense7)
dense9 = Dense(4)(dense8)
output1 = Dense(1)(dense9)

model = Model(inputs=input1, outputs=output1)

model.summary()

# model = Sequential()
# model.add(Dense(1000, input_dim=10, activation='relu'))
# model.add(Dense(60, activation='relu'))
# # model.add(Dense(40, activation='relu'))
# model.add(Dense(40, activation='relu'))
# model.add(Dense(40, activation='relu'))
# model.add(Dense(24, activation='relu'))
# model.add(Dense(24, activation='relu'))
# model.add(Dense(15, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(4, activation='relu'))
# model.add(Dense(2, activation='relu'))
# model.add(Dense(1))

# 3. compile
model.compile(loss='mse', optimizer='adam')

model.fit(x_train, y_train, epochs=100, batch_size=36, 
validation_split=0.03, shuffle=True)

# 4. evaluate, predict
loss = model.evaluate(x_test, y_test)
print('loss:', loss)
# loss: 3002.1416015625

y_pred = model.predict(x_test)
print('y예측값: ', y_pred)

r2 = r2_score(y_test, y_pred)
print('r2 스코어: ', r2)
# r2 스코어: 

# 과제 1
# r2 0.62 이상으로 올릴것 -> mail에 github 주소 보내도록
