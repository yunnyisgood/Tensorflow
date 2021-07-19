import numpy as np
import pandas as pd
from scipy.sparse import data
from sklearn import datasets
from sklearn.datasets import load_wine
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, PowerTransformer, QuantileTransformer
from sklearn.metrics import r2_score

# 다중분류, 0.8이상 완성

'''
./ -> 현재폴더
../ -> 상위폴더
'''

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


#sklearn 의 onehot 사용할것
one_hot_Encoder = OneHotEncoder()
one_hot_Encoder.fit(y)
y = one_hot_Encoder.transform(y).toarray()

print(np.unique(y)) # y라벨 확인

print(y.shape) # (4898, 7)로 바뀌어야 한다

x_train, x_test, y_train, y_test = train_test_split(x, y, 
test_size=0.7, random_state=9, shuffle=True)


scaler = PowerTransformer()
scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)


# modeling
model = Sequential()
model.add(Dense(5000, input_shape=(11, ), activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(7, activation='softmax')) 

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

# scaler 없을 때
# loss:  2.146963357925415
# accuracy:  0.47389909625053406

# minMax Scaler
# loss:  2.8873708248138428
# accuracy:  0.4234470725059509

# QuantileTransfomer
# loss:  1.7815757989883423
# accuracy:  0.4849810302257538

