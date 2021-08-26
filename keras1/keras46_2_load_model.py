import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler

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


x_train, x_test, y_train, y_test = train_test_split(x, y, 
train_size=0.7, shuffle=True, random_state=9)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_train = scaler.transform(x_train)

# 2. modeling
# model = load_model('./_save/keras46_1_save.model_1.h5')  # r2 스코어:  -3.5697476574048084
model = load_model('./_save/keras46_1_save.model_2.h5')  # r2 스코어:  -3.381975340046383
# -> model+ fit까지 모두 저장하였기 때문에 가중치까지 저장됨 => 결과치까지 저장된다
# 즉, 가중치까지 저장하려면 fit 이후인 위치에 save()를 해주어야 한다 

model.summary()


# 3. compile
'''
model.compile(loss='mse', optimizer='adam')

model.fit(x_train, y_train, epochs=100, batch_size=10, 
validation_split=0.03, shuffle=True)'''

# 4. evaluate, predict
loss = model.evaluate(x_test, y_test) # x_test를 통해 예측한 값, 실제 y_test의 값의 차이를 loss
print('loss:', loss)
# loss: 3002.1416015625

y_pred = model.predict(x_test)
print('y예측값: ', y_pred)

r2 = r2_score(y_test, y_pred)
print('r2 스코어: ', r2)
# r2 스코어: 

# 과제 1
# r2 0.62 이상으로 올릴것 -> mail에 github 주소 보내도록

# loss: 2337.067138671875
# r2 스코어:  0.5770988303739539

# load_model()s

