import numpy as np
import pandas as pd
from scipy.sparse import data
from sklearn import datasets
from sklearn.datasets import load_wine
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer, OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping
import time
from tensorflow.keras.utils import to_categorical


# 다중분류, 0.8이상 완성

datasets = pd.read_csv('../_data/winequality-white.csv', sep=';',
                        index_col=None, header=0)

print(datasets.shape) # (4898, 12)
print(datasets.info()) 
print(datasets.describe()) 

# 판다스 -> 넘파이로 변환 

datasets = datasets.to_numpy()

# x와 y를 분리
x = datasets[:, 0:11]
y = datasets[:, 11:]
# y= np.array(y).reshape(4898, )

print(x.shape) # (4898, 11)
print(y.shape) # (4898, 1)

# y = to_categorical(y)

one_hot_Encoder = OneHotEncoder()
one_hot_Encoder.fit(y)
y = one_hot_Encoder.transform(y).toarray()

print(np.unique(y)) # y라벨 확인

print(y.shape) # (4898, 7)로 바뀌어야 한다

print(y)
'''
x_train, x_test, y_train, y_test = train_test_split(x, y, 
test_size=0.7, random_state=9, shuffle=True)


scaler = QuantileTransformer()
scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)

print(x_train.shape, x_test.shape) # (1469, 11) (3429, 11)
print(y_train.shape, y_test.shape)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

print(y.shape) # (4898, 7)로 바뀌어야 한다



# modeling
model = Sequential()
model.add(LSTM(units=32, activation='relu', input_shape=(3, 1)))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(7))

model.summary()

# compile
model.compile(loss='categorical_crossentropy', optimizer='adam', 
metrics=['accuracy'])

es = EarlyStopping(monitor='loss', patience=20, mode='min', verbose=1)

hist = model.fit(x_train, y_train, epochs=1000, batch_size=8, validation_split=0.02,
callbacks=[es])

# evaluate

loss = model.evaluate(x_test, y_test)
print('loss: ', loss[0])
print('accuracy: ', loss[1])

y_pred = model.predict(x_test[:5])
# y_pred = model.predict(x_test[-5:-1])
print(y_test[:5])
# print(y_test[-5:-1])
print('--------softmax를 통과한 값 --------')
print(y_pred)
'''

# scaler 없을 때
# loss:  2.146963357925415
# accuracy:  0.47389909625053406

# minMax Scaler
# loss:  2.8873708248138428
# accuracy:  0.4234470725059509

# QuantileTransfomer
# loss:  1.7815757989883423
# accuracy:  0.4849810302257538

# CNN으로 실행했을 때 
# loss:  1.343643307685852
# accuracy:  0.4444444477558136

# DNN으로 실행했을 때 
# loss:  8.216516494750977
# accuracy:  0.2933799922466278