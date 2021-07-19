import numpy as np
from sklearn import datasets
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv1D, Flatten, Dropout, GlobalAveragePooling1D, MaxPool1D
from tensorflow.keras.callbacks import EarlyStopping
import time 


# 1. data 구성
datasets = load_boston()
x = datasets.data # (506, 13) input_dim = 13
y = datasets.target # (506,) output_dim = 1

x_train, x_test, y_train, y_test = train_test_split(x, y,
      test_size=0.2, shuffle=True, random_state=66)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)


# 2. model 구성

model = Sequential()
model.add(Conv1D(filters=32, kernel_size=2,                          
                        padding='same', activation='relu', input_shape=(13, 1))) 
model.add(Dropout(0.2))
model.add(Conv1D(32, 2, padding='same', activation='relu'))
model.add(MaxPool1D())

model.add(Conv1D(64, 2, padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv1D(64, 2, padding='same', activation='relu'))
model.add(MaxPool1D())

model.add(Conv1D(128, 2, padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv1D(128, 2, padding='same', activation='relu'))
model.add(MaxPool1D())

model.add(GlobalAveragePooling1D())
model.add(Dense(1))

# 3. 컴파일 훈련

model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1)

start_time = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=32, verbose=2,
    validation_split=0.015, callbacks=[es])
end_time = time.time() - start_time

# 4. 평가 예측

loss = model.evaluate(x_test, y_test)
print("time = ", end_time)
print('loss : ', loss)

y_predict = model.predict([x_test])

r2 = r2_score(y_test, y_predict)
print('R^2 score : ', r2)





