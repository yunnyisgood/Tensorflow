import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
import time
from tensorflow.keras.callbacks import EarlyStopping

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
model = Sequential()
model.add(Dense(64, input_dim=10, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

# model.save('./_save/keras46_1_save.model_1.hs')
model.save_weights('./_save/keras46_1_save_weights_1.h5')

model.summary()


# 3. compile

model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(monitor='val_loss', patience=8, verbose=1, mode='min')

start_time = time.time()
model.fit(x_train, y_train, epochs=50, validation_split=0.03, shuffle=True, batch_size=256)
end_time = time.time() - start_time

model.save_weights('./_save/keras46_1_save_weights_2.h5')


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

