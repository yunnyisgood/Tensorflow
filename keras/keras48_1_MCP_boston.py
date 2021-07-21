from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer, StandardScaler, MinMaxScaler
from sklearn.datasets import load_boston 
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import time 
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers.convolutional import Conv1D
from tensorflow.python.keras.saving.save import load_model

'''
과제 : 6개의 scaler에 대해 평가
'''

#1. data
datasets = load_boston()
x = datasets.data
y = datasets.target

print(x.shape) # (506, 13) -> 13가지의 특성 -> Input_dim = 13
print(y.shape) # (506,) -> output_dim = 1

x_train, x_test, y_train, y_test = train_test_split(x, y, 
train_size=0.7, shuffle=True, random_state=9)

print(x_train.shape, x_test.shape) # (354, 13) (152, 13)
print(y_train.shape, y_test.shape) # (354,) (152,)

# 전처리
# 2차원 -> 4차원

scaler = PowerTransformer()
scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

print(y_train.shape, y_test.shape) # (354,) (152,)


# modeling
'''model = Sequential()
model.add(LSTM(units=32, activation='relu', input_shape=(13, 1), return_sequences=True))
model.add(Conv1D(10,2, padding='same'))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

model.summary()


# compile
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

es = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
cp = ModelCheckpoint(monitor='val_loss', save_best_only=True, mode='auto', 
filepath='./_save/ModelCheckPoint/keras47_MCP_boston.hdf5')
start_time = time.time()
model.fit(x_train, y_train, epochs=100, verbose=1, callbacks=[es], validation_split=0.01,
shuffle=True, batch_size=10)
end_time = time.time() - start_time
'''
# model.save('./_save/ModelCheckPoint/keras47_MCP_boston.hdf5')
model = load_model('./_save/ModelCheckPoint/keras47_MCP_boston.hdf5')

# evaluate 
loss = model.evaluate(x_test, y_test)
# print("걸린시간: ", end_time)
print('loss: ', loss)

y_pred = model.predict(x_test)
print('x_test를 통한 y의 예측값:', y_pred)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred) # y_test, y_pred 차이
print('r2스코어:',r2)

'''
# PowerTransformer 전처리 이후
# loss: 17.08149528503418
# r2스코어: 0.807558585442023

CNN으로 실행
loss:  71.0339126586914
r2스코어: 0.19972649044261337

DNN으로 실행
loss:  25.253101348876953
r2스코어: 0.7154966059100147

Conv1d
걸린시간:  16.433225631713867
loss:  [26.68522834777832, 3.815570592880249]
r2스코어: 0.699362151102279

CheckPoint
걸린시간:  31.683058738708496
loss:  [26.76219367980957, 4.180410861968994]
r2스코어: 0.6984950530520678 -> 동일 
'''