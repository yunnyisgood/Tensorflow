import numpy as np
from sklearn import datasets
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
from tensorflow.keras.callbacks import EarlyStopping
import time

from tensorflow.python.keras.saving.save import load_model


datasets = load_breast_cancer() # (569, 30)

print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target

print(x.shape) # (569, 30)
print(y.shape) # (569,)

print(np.unique(y)) # y데이터는 0과 1데이터로만 구성되어 있다


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=9, shuffle=True)


scaler = MinMaxScaler()
scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)

print(x_train.shape, x_test.shape) # (398, 30) (171, 30)
print(y_train.shape, y_test.shape)


x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

'''
# modeling
model = Sequential()
model.add(LSTM(units=32, activation='relu', input_shape=(30, 1), return_sequences=True))
model.add(Conv1D(10, 2, activation='relu'))
model.add(Conv1D(10, 2, activation='relu'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

model.summary()

# compile
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# loss=mse가 아닌 binary_crossentropy -> 이진분류 방법 

es = EarlyStopping(monitor='loss', patience=20, verbose=1, mode='min')

cp = ModelCheckpoint(monitor='val_loss', mode='auto', save_best_only=True, 
filepath='./_save/ModelCheckPoint/keras47_MCP_cancer.hdf5')

hist = model.fit(x_train, y_train, epochs=1000, batch_size=8, validation_split=0.2, callbacks=[es])

# model.save('./_save/ModelCheckPoint/keras47_MCP_cancer.hdf5')'''

model = load_model('./_save/ModelCheckPoint/keras47_MCP_cancer.hdf5')

# evaluate
loss = model.evaluate(x_test, y_test) # evaluate는 metrics도 반환
print('loss: ', loss[0])
print('accuracy: ', loss[1])


'''
loss:  0.19166618585586548
accuracy:  0.9298245906829834

CNN으로 실행했을 때 
loss:  0.401380717754364
accuracy:  0.847953200340271

DNN으로 실행했을 때 
loss:  0.2741909623146057
accuracy:  0.9064327478408813

Conv1D로 실행했을 때 
loss:  9.832276344299316
accuracy:  0.3625730872154236

MCP
loss:  5.528964042663574
accuracy:  0.6374269127845764 -> 동일 
'''



