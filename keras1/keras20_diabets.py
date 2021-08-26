import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler, MinMaxScaler

'''
실습
1. loss 와 r2 평가
MinMax, Standard 스케일러 결과 명시
'''

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
train_size=0.8, shuffle=True, random_state=9)

# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)


# 2. modeling
model = Sequential()
model.add(Dense(512, input_dim=10, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# 3. compile
model.compile(loss='mse', optimizer='adam')

model.fit(x_train, y_train, epochs=100, batch_size=33, 
validation_split=0.03, shuffle=True)

# 4. evaluate, predict
loss = model.evaluate(x_test, y_test) 
print('loss:', loss)

y_pred = model.predict(x_test)
print('y예측값: ', y_pred)

r2 = r2_score(y_test, y_pred)
print('r2 스코어: ', r2)


# MinMaxScaler 전처리 이후
# loss: 2097.257080078125
# r2 스코어:  0.6139346339266243

# StandardScaler 전처리 이후
# loss: 2163.699462890625
# r2 스코어:  0.6024054583045837