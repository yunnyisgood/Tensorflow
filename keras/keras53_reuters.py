from re import T
from tensorflow.keras import datasets
from tensorflow.keras.datasets import reuters
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Conv1D, Flatten

'''
reuters 뉴스 데이터셋
46가지 토픽으로 라벨이 달린 11,228개의 로이터 뉴스로 이루어진 데이터셋.
'''

(x_train, y_train), (x_test, y_test) = reuters.load_data(
        num_words=10000, test_split=0.2
        # num_words=10000 매개변수는 데이터에서 가장 자주 등장하는 단어 10,000개로 제한
)

print(x_train[0], type(x_train[0])) #   <class 'list'>
print(y_train[0], type(y_test[0])) # 3 <class 'numpy.int64'>

print(type(x_train)) # <class 'numpy.ndarray'> -> 여러개의 리스트가 모여 넘파이로 구성된다 

print(len(x_train[0]), len(x_train[1])) # 87 56 -> list는 shape 안찍힌다

print(x_train.shape, x_test.shape) # (8982,) (2246,)
print(y_train.shape, y_test.shape) # (8982,) (2246,)

print("뉴스기사의 최대길이: ", max(len(i) for i in x_train))
# 뉴스기사의 최대길이:  2376
print("뉴스기사의 평균길이: ", sum(map(len, x_train))/ len(x_train))
# 뉴스기사의 평균길이:  145

# 전처리

x_train = pad_sequences(x_train,maxlen=100, padding='pre')
x_test = pad_sequences(x_test, maxlen=100, padding='pre')
print(x_train.shape, x_test.shape) # (8982, 100) (2246, 100)
print(type(x_train), type(x_test)) # <class 'numpy.ndarray'> <class 'numpy.ndarray'>

# y 확인
print(np.unique(x_train))
print(np.unique(y_train)) # 46개 

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train.shape, y_test.shape) # (8982, 46) (2246, 46)


# modeling
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=100, input_length=100))
model.add(LSTM(64, return_sequences=True))
model.add(Conv1D(64, 2))
model.add(Conv1D(16, 2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(46, activation='softmax'))

model.summary()

# compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

model.fit(x_train, y_train, epochs=100, batch_size=8)

# evaluate
acc = model.evaluate(x_test, y_test)
print('acc: ', acc)