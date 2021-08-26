from tensorflow.keras.datasets import imdb
import numpy as np
### 실습
# 1. 데이터
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
      # <class 'numpy.ndarray'>
print('뉴스기사의 최대길이 :', max(len(i) for i in x_train))      #2494
print('뉴스기사의 평균길이 :', sum(map(len, x_train)) / len(x_train))     #238.714
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
x_train = pad_sequences(x_train, maxlen=100, padding='pre')
x_test = pad_sequences(x_test, maxlen=100, padding='pre')
     # array([0, 1]  -> sigmoid, binary_crossentropy, acc
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape, x_test.shape,
      y_train.shape, y_test.shape)
    # y_train.shape: (25000, 2), y_test.shape: (25000, 2)
# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D, LSTM, Embedding
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=100, input_length=200))
model.add(LSTM(64, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(2, activation='sigmoid'))
model.summary()

# compile
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='min', patience=7, verbose=1)
import time
start = time.time()
model.fit(x_train, x_test, epochs=500, batch_size=512, validation_split=0.2, callbacks=[es])
end = time.time() - start
# 4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('걸린시간 :', end)
print('binary :', results[0])
print('acc :', results[1])

