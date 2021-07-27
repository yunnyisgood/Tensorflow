from tensorflow.keras.datasets import imdb
from tensorflow.keras import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Conv1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping


(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=10000
)

print(x_train[0])
print(y_train[0])

print(type(x_train), type(y_train))

print(x_train.shape, x_test.shape) # (25000,) (25000,)
print(y_train.shape, y_test.shape) # (25000,) (25000,)

print(max(len(i) for i in x_train)) # 2494
print(sum(map(len, x_train)) / len(x_train)) # 238

# 전처리
x_train = pad_sequences(x_train,maxlen=200, padding='pre')
x_test = pad_sequences(x_test, maxlen=200, padding='pre')
print(x_train.shape, x_test.shape) # (8982, 100) (2246, 100)
print(type(x_train), type(x_test)) # <class 'numpy.ndarray'> <class 'numpy.ndarray'>

# y 확인
print(np.unique(x_train))  # 10000
print(np.unique(y_train)) # 2

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape, x_test.shape,
      y_train.shape, y_test.shape)
# (25000,  100), (2500, 100), (25000,2)


# modeling
# model = Sequential()
# model.add(Embedding(input_dim=10000, output_dim=1000, input_length=200))
# model.add(LSTM(64, return_sequences=True, activation='relu'))
# model.add(Conv1D(64, 2, activation='relu'))
# model.add(Conv1D(16, 2, activation='relu'))
# model.add(Flatten())
# model.add(Dense(100, activation='relu'))
# # model.add(Dense(2, activation='sigmoid'))
# model.add(Dense(2, activation='softmax'))

model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=100, input_length=200))
model.add(LSTM(64, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(2, activation='sigmoid'))
model.summary()

# compile
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', patience=1, verbose=1, mode='auto', restore_best_weights=True)

model.fit(x_train, y_train, epochs=15, batch_size=256, callbacks=es, validation_split=0.2)

# evaluate
acc = model.evaluate(x_test, y_test)
print('loss: ', acc[0])
print('acc: ', acc[1])

'''
loss: 0.24146930
acc: 0.8755

loss:  0.38185542821884155
acc:  0.8753600120544434


'''
