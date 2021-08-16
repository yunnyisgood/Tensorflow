import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier # tree구조를 앙상블한 형태이다 
import warnings
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
from sklearn.pipeline import make_pipeline, Pipeline
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
warnings.filterwarnings('ignore')

dataset = pd.read_csv('../_data/open/train.csv', sep=',', index_col='index', 
header=0)
test =  pd.read_csv('../_data/open/test.csv', sep=',', index_col='index', 
header=0)
submission = pd.read_csv('../_data/open/sample_submission.csv')
stopwords = pd.read_csv('../_data/stopwords.txt').values.tolist()

print(dataset.shape, test.shape, submission.shape)
# (174304, 12) (43576, 11) (43576, 1)

#  data 컬럼 numpy로 변환
# train_text1 = np.array(dataset['내역사업명'].tolist())
train_text2 = np.array(dataset['사업_부처명'].tolist())
train_text3 = np.array(dataset['요약문_한글키워드'].tolist())

# test_text1 = np.array(test['내역사업명'].tolist())
test_text2 = np.array(test['사업_부처명'].tolist())
test_text3 = np.array(test['요약문_한글키워드'].tolist())



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
train_text = np.array(temp_list)
pred_text = np.array(temp_list2)

train_text = np.concatenate(train_text, train_text2, train_text3)
pred_text = np.concatenate(pred_text, test_text2, test_text3)

print(train_text)

# train_text = np.stack(( train_text2, train_text3, train_text), axis=0)
# pred_text = np.stack(( test_text2, test_text3, pred_text), axis=0)

# print(train_text)

np.save('../_data/tain_text1.npy', arr=train_text)
np.save('../_data/labels1.npy', arr=labels)
np.save('../_data/pred_text1.npy', arr=pred_text)

# print(labels.shape, train_text.shape, pred_text.shape)
# # (174304,) (4, 174304) (4, 43576)

train_text = np.load('../_data/tain_text.npy', allow_pickle = True).tolist()
labels = np.load('../_data/labels.npy', allow_pickle = True).tolist()
pred_text = np.load('../_data/pred_text.npy', allow_pickle = True).tolist()


# print(train_text)
# print(type(train_text))


# print('len(train_text): ',len(train_text))
# print(train_text[:10])
# # ['유전', '정보', '를', '활용', '한', '새로운', '해충', '분류군', '동정', '기술']


# 벡터화
vectorizer = CountVectorizer(tokenizer = lambda x: x, lowercase=False)
train_features=vectorizer.fit_transform(train_text)
test_features=vectorizer.transform(pred_text)

# print('len(train_features): ',len(train_features))
print(type(train_features))
# <class 'scipy.sparse.csr.csr_matrix'>

# reshape
train_features = train_features.reshape(174304, 4)
test_features = test_features.reshape(43576, 4)


x_train, x_test, y_train, y_test = train_test_split(
    train_features, labels, test_size=0.2, random_state=42)

print(x_train.shape, y_train.shape)

# n_splits = 5
# kfold = KFold(n_splits=n_splits,  shuffle=True, random_state=66)


# #modeling

# # parameters = [

# #     {"xgb__n_estimators":[100, 200, 300], "xgb__learning_rate": [0.1, 0.3, 0.001, 0.01],
# #     "xgb__max_depth": [4,5,6]},
# #     {"xgb__n_estimators":[90, 100, 110], "xgb__learning_rate": [0.1,  0.001, 0.01],
# #     "xgb__max_depth": [4,5,6], "xgb__colsample_bytree":[0.6, 0.9, 1]},
# #      {"xgb__n_estimators":[90, 110], "xgb__learning_rate": [0.1,  0.001, 0.5],
# #     "xgb__max_depth": [4,5,6], "xgb__colsample_bytree":[0.6, 0.9, 1],
# #     "xgb__colsample_bylevel": [0.6, 0.7, 0.9]}

# # ]

# # model = make_pipeline(MaxAbsScaler(), RandomForestClassifier())
# # pipeline을 사용하여 scaling, modeling을 한번에

# #2.modeling
# # model = make_pipeline(MaxAbsScaler(), SVC())
# # pipeline을 사용하여 scaling, modeling을 한번에

# #3. training
# # model.fit(x_train, y_train)



# parameters = [
#     {'rf__min_samples_leaf':[3, 5, 7], 'rf__max_depth':[2, 3, 5, 10]},
#     { 'rf__min_samples_split':[6, 8, 10]}
# ]

# pipe = Pipeline([("scaler", MaxAbsScaler()), ("rf", RandomForestClassifier())])

# model = RandomizedSearchCV(pipe, parameters, cv=kfold, verbose=1)

# #3. training
# model.fit(x_train, y_train)

# #4.predict
# print("model.score: ", model.score(x_test, y_test))

# y_pred  = model.predict(test_features)
# print('1',y_pred[:10])
# # y_pred = np.argmax(y_pred, axis=1)
# # print('2', y_pred[:10])
# # IndexError: invalid index to scalar variable.

# print(model.feature_importances_)

# submission = pd.read_csv('../_data/open/sample_submission.csv')

# submission['label'] = y_pred
# # ic(submission.shape)

# submission.to_csv('../_data/rf_baseline.csv', index=False)


# date_time = str(datetime.now())
# date_time = date_time[:date_time.rfind(':')].replace(' ', '_')
# date_time = date_time.replace(':','시') + '분'

# folder_path = os.getcwd()
# csv_file_name = 'weather_최종제출{}.csv'.format(date_time)

# submission.to_csv(csv_file_name)

'''
컬럼 삭제 전 

model = make_pipeline(MaxAbsScaler(), RandomForestClassifier())
model.score:  0.9208571182696996

model = make_pipeline(MaxAbsScaler(), DecisionTreeClassifier())
model.score:  0.9100427411720834

model = make_pipeline(MaxAbsScaler(), SVC())


컬럼 삭제 후

'''