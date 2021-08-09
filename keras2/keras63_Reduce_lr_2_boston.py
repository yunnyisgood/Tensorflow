from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer, StandardScaler, MinMaxScaler
from sklearn.datasets import load_boston 
from tensorflow.python.keras.models import Sequential 
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import time
'''
과제 : 6개의 scaler에 대해 평가
'''

#1. data
datasets = load_boston()
x = datasets.data
y = datasets.target

print(x.shape) # (506, 13) -> 13가지의 특성 -> Input_dim = 13
print(y.shape) # (506,) -> output_dim = 1

print(datasets.feature_names)

x_train, x_test, y_train, y_test = train_test_split(x, y, 
train_size=0.7, shuffle=True, random_state=9)

# scaler = PowerTransformer()
# scaler.fit(x_train)
# scaler.transform(x_train)
# scaler.transform(x_test)

# 2.model
model = Sequential()
model.add(Dense(1000, input_dim=13, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))


# 3. compile
optimizer = Adam(lr=0.001)

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

es = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto', verbose=1, factor=0.5)

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=100, verbose=1, callbacks=[es, reduce_lr], validation_split=0.2,
shuffle=True, batch_size=8)
end_time = time.time() - start_time


# 4. evaluate and predict
loss = model.evaluate(x_test, y_test)
print('loss:', loss)

# loss: 37.513221740722656
# loss: 17.039806365966797

y_pred = model.predict(x_test)
print('x_test를 통한 y의 예측값:', y_pred)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred) # y_test, y_pred 차이
print('r2스코어:',r2)

'''
PowerTransformer, MinMaxScaler, StandardScaler 순서로
가장 성능이 우수
 
'''

# r2스코어: 0.3354429571900964
# r2스코어: 0.7853412907684698

# MinMaxScaler 전처리 이후
# loss: 17.463186264038086
# r2스코어: 0.8032583968614756

# StandardScaler 전처리 이후
# loss: 18.29416847229004
# r2스코어: 0.793896502480951

# MaxAbsScaler 전처리 이후
# loss: 25.627376556396484
# r2스코어: 0.7112800084011388

# RobustScaler 전처리 이후
# loss: 22.079360961914062
# r2스코어: 0.751252205269121

# QuantileTransformer 전처리 이후
# loss: 25.287765502929688
# r2스코어: 0.7151061161907877

# PowerTransformer 전처리 이후
# loss: 17.08149528503418
# r2스코어: 0.807558585442023

# Adam(lr=0.001)
# 2.7945170586463064e-06
# r2스코어: -1.3656342712278744


