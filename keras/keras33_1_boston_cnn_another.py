from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer, StandardScaler, MinMaxScaler
from sklearn.datasets import load_boston 
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPool1D, Dropout, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import time 
import matplotlib.pyplot as plt

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

scaler = PowerTransformer()
scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)

# 전처리
# 2차원 -> 3차원

x_train = x_train.reshape(354, 13, 1)
x_test = x_test.reshape(152, 13, 1)

print(y_train.shape, y_test.shape) # (354,) (152,)


# modeling
model = Sequential()
model.add(Conv1D(filters=6, kernel_size=2, 
                    padding='same', input_shape=(13,1), activation='relu'))
# model.add(Dropout(0, 2)) # 20%의 드롭아웃의 효과를 낸다 
model.add(Dropout(0.2))
model.add(Conv1D(16, 2, padding='same', activation='relu'))   
model.add(MaxPool1D())

model.add(Conv1D(64, 2,padding='same', activation='relu'))  
model.add(Dropout(0.2))
model.add(Conv1D(64, 2, padding='same', activation='relu')) 
model.add(MaxPool1D())


model.add(Conv1D(128, 2, padding='same', activation='relu')) 
model.add(Dropout(0.2))
model.add(Conv1D(128, 2, padding='same', activation='relu')) 
model.add(MaxPool1D())

# 여기까지가 convolutional layer 

model.add(GlobalAveragePooling1D())
model.add(Dense(1))



# compile
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')

start_time = time.time()
model.fit(x_train, y_train, epochs=100, verbose=1, callbacks=[es], validation_split=0.01,
shuffle=True, batch_size=10)
end_time = time.time() - start_time


# evaluate 
loss = model.evaluate(x_test, y_test)
print("걸린시간: ", end_time)
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
loss:  [71.0339126586914, 5.953949928283691]
r2스코어: 0.19972649044261337


'''