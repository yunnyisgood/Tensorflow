import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.layers import Dense

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
train_size=0.7, shuffle=True, random_state=66)

# 2. modeling
model = Sequential()
model.add(Dense(128, input_dim=10, activation='relu'))
# 활성화 함수 -> 모든 layer에 존재 
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1)) # 일단 마지막layer에는 사용하지 않는다
# 그다음 layer에 전달하는 값들을 한정시킨다.


# 3. compile
model.compile(loss='mse', optimizer='adam')

model.fit(x_train, y_train, epochs=100, batch_size=45, 
validation_split=0.2)

# 4. evaluate, predict
loss = model.evaluate(x_test, y_test)
print('loss:', loss)
# loss: 3002.1416015625


y_pred = model.predict(x_test)
print('y예측값: ', y_pred)

r2 = r2_score(y_test, y_pred)
print('r2 스코어: ', r2)
# r2 스코어:  0.5065703468309615

# 과제 1
# r2 0.62 이상으로 올릴것 -> mail에 github 주소 보내도록
