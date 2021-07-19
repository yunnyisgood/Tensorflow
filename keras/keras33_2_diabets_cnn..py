import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, GlobalAveragePooling2D
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
from tensorflow.python.ops.gen_array_ops import QuantizeAndDequantize
from tensorflow.keras.callbacks import EarlyStopping
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

print(x_train.shape, x_test.shape) # (353, 10) (89, 10)
print(y_train.shape, y_test.shape)

scaler =  QuantileTransformer()
scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)

x_train = x_train.reshape(353, 10, 1, 1)
x_test = x_test.reshape(89, 10, 1, 1)


# modeling
model = Sequential()
model.add(Conv2D(filters=6, kernel_size=(1,1), 
                    padding='same', input_shape=(1, 1, 1), activation='relu'))
# model.add(Dropout(0, 2)) # 20%의 드롭아웃의 효과를 낸다 
model.add(Dropout(0.2))
model.add(Conv2D(16, (1,1), padding='same', activation='relu'))   

model.add(Conv2D(64, (1,1),padding='valid', activation='relu'))  
model.add(Dropout(0.2))
model.add(Conv2D(64, (1,1), padding='same', activation='relu')) 


model.add(Conv2D(128, (1,1), padding='valid', activation='relu')) 
model.add(Dropout(0.2))
model.add(Conv2D(128, (1,1), padding='same', activation='relu')) 

# 여기까지가 convolutional layer 

model.add(GlobalAveragePooling2D())
model.add(Dense(1))



# compile
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

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
# print('y예측값: ', y_pred)

r2 = r2_score(y_test, y_pred)
print('r2 스코어: ', r2)
# r2 스코어:  0.5881755653170027


'''
QuantileTransformer, MinMaxScaler 순서로 성능 우수


QuantileTransformer 전처리 이후
loss: 2079.048828125
r2 스코어:  0.6179605805519554

CNN으로 실행
loss:  [3862.361572265625, 51.89512252807617]
r2 스코어:  0.29026469959891266

'''