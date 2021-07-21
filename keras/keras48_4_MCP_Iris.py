import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
from tensorflow.keras.callbacks import EarlyStopping
import time

# 다중분류, One-Hot-Encoding 

datasets = load_iris()
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target

print(x.shape)  # (150, 4)
print(y.shape)  # (150,)
print(y) # -> cancer와 마찬가지로 0, 1로 구성되어있다

''' 
One-Hot-Encoding (150, ) -> (150, 3, 1)
0 -> [1, 0, 0]
1 -> [0, 1, 0]
2 -> [0, 0, 1]

ex) [0, 1, 2, 1]
=> [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]]
=> (4, )에서 (4, 3)으로 바뀌었다

-> 원래 y 데이터 (150, )에서 (150, 3)으로 바뀌게 됨을 알 수 있다
'''
y = to_categorical(y) # -> one-Hot-Encoding 하는 방법
print(y[:5])

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=9, shuffle=True)


scaler = MinMaxScaler()
scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)

print(x_train.shape, x_test.shape) # (105, 4) (45, 4)
print(y_train.shape, y_test.shape) # (105, 3) (45, 3)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

'''
# modeling
model = Sequential()
model.add(LSTM(units=32, activation='relu', input_shape=(4, 1), return_sequences=True))
model.add(Conv1D(10, 2, activation='relu'))
model.add(Conv1D(10, 2, activation='relu'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(3))

model.summary()

# compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# loss=mse가 아닌 binary_crossentropy -> 이진분류 방법 

es = EarlyStopping(monitor='loss', patience=20, verbose=1, mode='min')
cp = ModelCheckpoint(monitor='val_loss', mode='auto', save_best_only=True, 
filepath='./_save/ModelCheckPoint/keras47_MCP_Iris.hdf5')

hist = model.fit(x_train, y_train, epochs=1000, batch_size=8, validation_split=0.2, callbacks=[es])

model.save('./_save/ModelCheckPoint/keras47_MCP_Iris.hdf5')'''

model = load_model('./_save/ModelCheckPoint/keras47_MCP_Iris.hdf5')

# evaluate
loss = model.evaluate(x_test, y_test) # evaluate는 metrics도 반환
print('loss: ', loss[0])
print('accuracy: ', loss[1])

y_pred = model.predict(x_test[:5])
print(y_test[:5])
print('--------softmax를 통과한 값 --------')
print(y_pred)

'''
loss:  0.10622021555900574
accuracy:  0.9555555582046509

CNN으로 실행했을 떄
loss:  0.592060923576355
accuracy:  0.7555555701255798

DNN으로 실행했을 떄 
loss:  10.745397567749023
accuracy:  0.4000000059604645

Conv1D
loss:  6.447238445281982
accuracy:  0.42222222685813904

MCP
loss:  9.670857429504395
accuracy:  0.4000000059604645
'''