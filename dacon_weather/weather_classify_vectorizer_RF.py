import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier # tree구조를 앙상블한 형태이다 
import warnings
from sklearn.pipeline import make_pipeline, Pipeline
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
import time
import matplotlib.pyplot as plt
import datetime
from tensorflow.python.keras.layers.core import Dropout
import re
from icecream import ic
import os
from tensorflow.keras.optimizers import Adam, Adagrad, Adadelta, Adamax
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from konlpy.tag import Okt
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import CountVectorizer
from datetime import datetime
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
warnings.filterwarnings('ignore')


dataset = pd.read_csv('../_data/open/train.csv', sep=',', index_col='index', 
header=0)
test =  pd.read_csv('../_data/open/test.csv', sep=',', index_col='index', 
header=0)
submission = pd.read_csv('../_data/open/sample_submission.csv')
stopwords = pd.read_csv('../_data/stopwords.txt').values.tolist()

print(dataset.shape, test.shape, submission.shape)
# (174304, 12) (43576, 11) (43576, 1)


# 형태소화
okt = Okt()
temp_list = []
for sentence in dataset['과제명']:
    temp = []
    temp = re.sub('[a-zA-z]','',sentence)
    temp = re.sub('[-=+,#○▷▶■ ●□/\?:^$.@*\"*~&%`!\\`|\(\)\[\]\<\>`\'...>]', '', str(sentence)) 
    temp = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]", " ", str(sentence))
    temp  = okt.morphs(sentence) #  문장에서 명사 추출 
    temp = [word for word in temp if not word in stopwords]  # 불용어 처리 
    temp_list.append(temp)

temp_list2 = []
for sentence in test['과제명']:
    temp = []
    temp = re.sub('[a-zA-z]','',sentence)
    temp = re.sub('[-=+,#○▷▶■ ●□/\?:^$.@*\"*~&%`!\\`|\(\)\[\]\<\>`\'...>]', '', str(sentence)) 
    temp = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]", " ", str(sentence))
    temp  = okt.morphs(sentence) #  문장에서 명사 추출 
    temp = [word for word in temp if not word in stopwords]  # 불용어 처리 
    temp_list2.append(temp)


# x, y 개수를 맞춰주기 위해 다시 비어있는 행을 삭제한다 
# dataset = dataset.dropna(axis=0)
# test = test.dropna(axis=0)

print(dataset.shape, test.shape)
# (171138, 12) (43576, 11)

print('len(temp_list): ',len(temp_list))
# len(temp_list):  174304
print(temp_list[:10])

labels = np.array(dataset['label'])
train_text = temp_list
pred_text = temp_list2


print('len(train_text): ',len(train_text))
print(train_text[:10])
# ['유전', '정보', '를', '활용', '한', '새로운', '해충', '분류군', '동정', '기술']

# 벡터화
vectorizer = CountVectorizer(tokenizer = lambda x: x, lowercase=False)
train_features=vectorizer.fit_transform(train_text)
test_features=vectorizer.transform(pred_text)

# print('len(train_features): ',len(train_features))
print(type(train_features))
# <class 'scipy.sparse.csr.csr_matrix'>

'''
TypeError: MinMaxScaler does not support sparse input. Consider using MaxAbsScaler instead.
'''

x_train, x_test, y_train, y_test = train_test_split(
    train_features, labels, test_size=0.2, random_state=42)


#modeling
model = make_pipeline(MaxAbsScaler(), RandomForestClassifier())
# pipeline을 사용하여 scaling, modeling을 한번에

#3. training
model.fit(x_train, y_train)

#4.predict
print("model.score: ", model.score(x_test, y_test))

y_pred  = model.predict(test_features)
print('1',y_pred[:10])
# y_pred = np.argmax(y_pred, axis=1)
# print('2', y_pred[:10])
# IndexError: invalid index to scalar variable.

submission['label'] = y_pred
# ic(submission.shape)



# Modeling
'''model = Sequential()
model.add(Embedding(input_dim=1000, output_dim=1000, input_length=40))
model.add(GlobalAveragePooling1D())
model.add(Dense(128, activation='relu'))
model.add(Dense(46, activation='softmax'))


optimizer = Adam(lr=0.001)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['f1_score'])     

es = EarlyStopping(monitor='val_loss', mode='auto', patience=1, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=1, mode='auto', verbose=2, factor=0.1)

model.fit(train_inputs, labels, epochs=30, callbacks=[es, reduce_lr], validation_split=0.2)


# Predict
y_predict = model.predict(test_inputs)
y_predict = np.argmax(y_predict, axis=1)
f1_score = f1_score(labels, y_predict)
print('f1_score: ', f1_score)

submission['label'] = y_predict
# ic(submission.shape)
ic(y_predict)


date_time = str(datetime.now())
date_time = date_time[:date_time.rfind(':')].replace(' ', '_')
date_time = date_time.replace(':','시') + '분'

folder_path = os.getcwd()
csv_file_name = 'weather_최종제출{}.csv'.format(date_time)

submission.to_csv(csv_file_name)'''