import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Embedding, LSTM, Conv1D, Flatten, MaxPooling1D, Conv2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
import time
import matplotlib.pyplot as plt
import datetime
from tensorflow.python.keras.layers.core import Dropout
import re
from sklearn.model_selection import StratifiedKFold


dataset = pd.read_csv('./dacon/_data/train_data.csv', sep=',', index_col='index', 
header=0)
x_pred =  pd.read_csv('./dacon/_data/test_data.csv', sep=',', index_col='index', 
header=0)
submission = pd.read_csv('./dacon/_data/sample_submission.csv', sep=',', index_col='index', 
header=0)

print(dataset.columns)
print(x_pred)

def clean_text(text):# 특수 문자 제거 
    cleaned_tex = re.sub('[-=+,#/\?:^$.@*\"*~&%`!\\`|\(\)\[\]\<\>`\'...>]', '', text) 
    cleaned_text = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]", " ", text)
    return cleaned_text

x = dataset['title'].dropna(axis=0).apply(lambda x : clean_text(x))
y = dataset['topic_idx'].dropna(axis=0)
x_pred = x_pred['title'].dropna(axis=0).apply(lambda x : clean_text(x))

# 판다스 -> 넘파이로 변환 
x = x.to_numpy()
y = y.to_numpy()
x_pred = x_pred.to_numpy()

# Token
token = Tokenizer(num_words=1000)
token.fit_on_texts(x)
print('token.word_index:', token.word_index)
x = token.texts_to_sequences(x)
print('x:', x[:10])

token2 = Tokenizer(num_words=1000)
token2.fit_on_texts(x_pred)
print('token2.word_index: ', token2.word_index)
x_pred = token2.texts_to_sequences(x_pred)

word_size = len(token.word_index)
print('word_size: ',word_size) # 101081


print("문장  최대길이: ", max(len(i) for i in x))
# 9
print("문장 평균길이: ", sum(map(len, x)) / len(x))
# 2

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=9)

print('1: ', x_train[:10])
# 전처리

x_train = pad_sequences(x_train, maxlen=6, padding='pre')
x_test = pad_sequences(x_test, maxlen=6, padding='pre')
x_pred = pad_sequences(x_pred, maxlen=6, padding='pre')

print('2: ', x_train[:10])

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape, x_test.shape,
      y_train.shape, y_test.shape, x_pred.shape)
# (31957, 2) (13697, 2) (31957, 7) (13697, 7) (9131, 2)

# modeling 
# model = Sequential()
# model.add(Embedding(input_dim=15000, output_dim=1000, input_length=6))
# model.add(Flatten())
# model.add(Dense(128, activation = "relu"))
# model.add(Dropout(0.8))
# model.add(Dense(7, activation = "softmax"))


model = Sequential()
model.add(Embedding(input_dim=1000, output_dim=1000, input_length=6))
model.add(LSTM(16, activation='relu'))
model.add(Dense(7, activation='softmax'))

model.summary()

# compile 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto', restore_best_weights=True)

start_time = time.time()

# 계층 교차 검증
n_fold = 5  
seed = 9
cv = StratifiedKFold(n_splits = n_fold, shuffle=True, random_state=seed)

for i, (i_trn, i_val) in enumerate(cv.split(x_train, y_train), 1):
    print(f'training model for CV #{i}')

    hist = model.fit(x_train[i_trn], 
            to_categorical(y_train[i_trn]),
            validation_data=(x_train[i_val], to_categorical(y_train[i_val])),
            epochs=100,
            batch_size=1024)
# hist = model.fit(x_train, y_train, epochs=100, batch_size=128, verbose=1, 
# callbacks=es, validation_split=0.2 )
end_time = time.time() -start_time

date = datetime.datetime.now()
date_time = date.strftime("%m%d_%H%M")
weight_path = './dacon/_save/' 
weight_path = "".join([weight_path, "news_topic_weight_save", date_time, ".h5"])   

model.save_weights(weight_path)

# evaluate
results = model.evaluate(x_test, y_test)
print("걸린시간: ", end_time)
print('loss: ',results[0])
print('acc: ',results[1])

csv_path = './dacon/_save/' 
csv_path = "".join([csv_path, "news_topic__submission", date_time, ".csv"])

y_pred = model.predict(x_pred)
y_pred = np.argmax(y_pred, axis=1)
submission['topic_idx'] = y_pred
submission.to_csv(csv_path)

# 시각화 
plt.figure(figsize=(9,5))

# 1
plt.subplot(2, 1, 1) 
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

# 2
plt.subplot(2, 1, 2) 
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.grid()
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['acc', 'val_acc'])

plt.show()


'''
걸린시간:  87.26695418357849
loss:  0.9827284812927246
acc:  0.6602175831794739
'''