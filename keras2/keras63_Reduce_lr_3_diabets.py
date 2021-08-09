import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import time


'''
과제 : 6개의 scaler에 대해 평가
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
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler =  RobustScaler()
scaler =  PowerTransformer()
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
optimizer = Adam(lr=0.001)

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

es = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto', verbose=1, factor=0.5)

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=1000, verbose=1, callbacks=[es, reduce_lr], validation_split=0.2,
shuffle=True, batch_size=8)
end_time = time.time() - start_time

# 4. evaluate, predict
loss = model.evaluate(x_test, y_test) 
print('loss:', loss)
# loss: 2241.13818359375

y_pred = model.predict(x_test)
# print('y예측값: ', y_pred)

r2 = r2_score(y_test, y_pred)
print('r2 스코어: ', r2)
# r2 스코어:  0.5881755653170027


'''
QuantileTransformer, MinMaxScaler 순서로 성능 우수

'''

# MinMaxScaler 전처리 이후
# loss: 2097.257080078125
# r2 스코어:  0.6139346339266243

# StandardScaler 전처리 이후
# loss: 2163.699462890625
# r2 스코어:  0.6024054583045837

# MaxAbsScaler 전처리 이후
# loss: 2173.5859375
# r2 스코어:  0.6005887828234024

# RobustScaler 전처리 이후
# loss: 2341.55224609375
# r2 스코어:  0.56972377635531

# QuantileTransformer 전처리 이후
# loss: 2079.048828125
# r2 스코어:  0.6179605805519554

# PowerTransformer 전처리 이후
# loss: 2284.801025390625
# r2 스코어:  0.5801522006661404

# Adam(lr=0.001)
# loss: [1.7262578694499098e-05
# r2 스코어:  -3.853453236327203