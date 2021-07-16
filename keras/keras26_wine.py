import numpy as np
from sklearn.datasets import load_wine
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, PowerTransformer, QuantileTransformer


'''
완성
accuracy 0.8 이상 만들어 볼 것
=> 0.9279999732971191 가 현재까지 최대
'''

datasets = load_wine()

print(type(datasets))  # <class 'sklearn.utils.Bunch'>
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target

print(x.shape) # (178, 13)
print(y.shape) # (178, )

y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, 
test_size=0.7, random_state=9, shuffle=True)

scaler = QuantileTransformer()
scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)

# modeling
model = Sequential()
model.add(Dense(1000, input_shape=(13, ), activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax')) 

# compile
model.compile(loss='categorical_crossentropy', optimizer='adam', 
metrics=['accuracy'])

es = EarlyStopping(monitor='loss', patience=20, mode='min', verbose=1)

hist = model.fit(x_train, y_train, epochs=1000, batch_size=8, validation_split=0.2,
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

# 범위 -> [:5]

# scaler 없이
# loss:  0.750230073928833
# accuracy:  0.671999990940094


# MinMax scaler 사용할 때
# loss:  0.5610019564628601
# accuracy:  0.7839999794960022

# -> MinMax Scaler인데 node 조정 후
# loss:  0.24176639318466187
# accuracy:  0.9279999732971191

# PowerTransformer
# loss:  0.8162326812744141
# accuracy:  0.6320000290870667

# QuantileTransformer
# loss:  0.6085973381996155
# accuracy:  0.6800000071525574

# 범위 -> [-5: -1]
# loss:  0.6953398585319519
# accuracy:  0.6320000290870667