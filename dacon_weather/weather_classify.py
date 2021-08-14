import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
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
from icecream import ic
from datetime import datetime
import os
from tensorflow.keras.optimizers import Adam, Adagrad, Adadelta, Adamax
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

submission = pd.read_csv('../_data/open/sample_submission.csv', sep=',', index_col='index', 
header=0)


'''dataset = pd.read_csv('../_data/open/train.csv', sep=',', index_col='index', 
header=0)
x_pred =  pd.read_csv('../_data/open/test.csv', sep=',', index_col='index', 
header=0)
submission = pd.read_csv('../_data/open/sample_submission.csv', sep=',', index_col='index', 
header=0)

print(dataset.columns)
print(dataset)

# x = dataset[['사업명','사업_부처명', '요약문_연구목표','요약문_연구내용',
# '요약문_기대효과','요약문_한글키워드']]
# y = dataset[['label']]
x_pred = x_pred [['사업명','사업_부처명', '요약문_연구목표','요약문_연구내용',
'요약문_기대효과','요약문_한글키워드']]

def clean_text(text):# 특수 문자 제거 
    # cleaned_text = re.sub('[-=+,#/\?:^$.@*\"*~&%`!\\`|\(\)\[\]\<\>`\'...>]', '', str(text)) 
    cleaned_text = re.sub('[a-zA-z]','',text)
    cleaned_tex = re.sub('[-=+,#○▷▶■ ●□/\?:^$.@*\"*~&%`!\\`|\(\)\[\]\<\>`\'...>]', '', str(text)) 
    cleaned_text = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]", " ", str(text))
    return cleaned_text

dataset[['사업명','사업_부처명', '요약문_연구목표','요약문_연구내용',
 '요약문_기대효과','요약문_한글키워드']] = dataset[['사업명','사업_부처명', '요약문_연구목표','요약문_연구내용', '요약문_기대효과','요약문_한글키워드']].dropna(axis=0).applymap(lambda x : clean_text(x))
dataset[['label']] = dataset[['label']].dropna(axis=0)
x_pred = x_pred.dropna(axis=0).applymap(lambda x : clean_text(x))


# x, y 개수를 맞춰주기 위해 다시 비어있는 행을 삭제한다 
dataset = dataset.dropna(axis=0)

x = dataset[['사업명','사업_부처명', '요약문_연구목표','요약문_연구내용',
'요약문_기대효과','요약문_한글키워드']]
y = dataset[['label']]
x_pred = x_pred [['사업명','사업_부처명', '요약문_연구목표','요약문_연구내용',
'요약문_기대효과','요약문_한글키워드']]

print(x.shape, y.shape, x_pred.shape)
# (171138, 6) (171138, 1) (42805, 6)


# 판다스 -> 넘파이로 변환 
x = x.to_numpy()
y = y.to_numpy()
x_pred = x_pred.to_numpy()


np.save('../_data/x.npy', arr=x)
np.save('../_data/y.npy', arr=y)
np.save('../_data/x_pred.npy', arr=x_pred)'''

x = np.load('../_data/x.npy', allow_pickle = True).tolist()
y = np.load('../_data/y.npy', allow_pickle = True).tolist()
x_pred = np.load('../_data/x_pred.npy', allow_pickle = True).tolist()

# print(x.shape, y.shape, x_pred.shape)
# (171138, 6) (171138, 1) (42805, 6)
print(x[:10])

print(y[:10])

# 벡터화
vectorizer = TfidfVectorizer(analyzer='char_wb', sublinear_tf=True, ngram_range=(1, 2), max_features=70000, binary=False)

x = vectorizer.fit_transform(x).astype('float32')
x_pred = vectorizer.transform(x_pred).astype('float32')


# Modeling
model = Sequential()
model.add(Dense(1000, input_dim=70000, activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(45, activation='softmax'))


optimizer = Adam(lr=0.001)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['acc'])     # SGD:경사하강법

es = EarlyStopping(monitor='val_loss', mode='auto', patience=1, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=1, mode='auto', verbose=1, factor=0.1)

model.fit(x[:70000], y[:70000], epochs=6, batch_size=3000, callbacks=[es, reduce_lr], validation_data=(x[:70000], y[:70000]))


# Predict
y_predict = model.predict(x_pred)
y_predict = np.argmax(y_predict, axis=1)


submission['label'] = y_predict
# ic(submission.shape)
ic(y_predict)


date_time = str(datetime.now())
date_time = date_time[:date_time.rfind(':')].replace(' ', '_')
date_time = date_time.replace(':','시') + '분'

folder_path = os.getcwd()
csv_file_name = 'weather_최종제출{}.csv'.format(date_time)

submission.to_csv(csv_file_name)