'''
Data  증폭
-> Image Generator

Smote: 데이터의 개수가 적은 클래스의 표본을 가져온 뒤 임의의 값을 추가하여 
새로운 샘플을 만들어 데이터에 추가하는 오버샘플링 방식

'''

from imblearn.over_sampling import SMOTE
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
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, PowerTransformer, QuantileTransformer, StandardScaler, RobustScaler
from sklearn.metrics import r2_score
from xgboost import XGBClassifier
import warnings
import time
warnings.filterwarnings('ignore')

datasets = load_wine()
x = datasets.data
y = datasets.target

print(x.shape, y.shape) #(178, 13) (178,)

print(pd.Series(y).value_counts())
# 1    71
# 0    59
# 2    48

print(y)
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1  
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1  
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2  
#  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]

x_new = x[:-30]
y_new = y[:-30]
print(x_new.shape, y_new.shape) # (148, 13) (148,)
print(pd.Series(y_new).value_counts())
# 1    71
# 0    59
# 2    18

x_train,x_test, y_train, y_test =  train_test_split(
    x_new, y_new, train_size=0.75, shuffle=True, random_state=66, 
    stratify=y_new # trian, test size의 비율에 맞게 라벨값이 분리되어 진다 
)
# train_size = 0.75가 기본값

print(pd.Series(y_train).value_counts())

'''
=> train_size 뿐만 아니라 random_state에 따라서도 계속 라벨 분리 비율이 달라짐
=> stratify를 사용하여 train size에 따라 분리되도록 해준다
train_size=0.75
1    53
0    44 -> 9개 늘려주기 
2    14 -> 39개 증폭
=> 총 데이터가 159개가 되도록
'''

model = XGBClassifier(n_jobs=-1)

model.fit(x_train, y_train, eval_metric='mlogloss')

score = model.score(x_test, y_test)
print('score : ', score)
# score :  0.9459459459459459

################### SMOTE 적용 #####################
print("===============SMOTE적용================")

smote = SMOTE(random_state=66)

x_smote_train, y_smote_train = smote.fit_resample(x_train, y_train)
# smote가 적용된 x_trian, y_train 구하기

print(pd.Series(y_smote_train).value_counts())
print(x_smote_train.shape, y_smote_train.shape)
# (159, 13) (159,) => 159개 데이터로 증폭

print("smote 전: ", x_train.shape, y_train.shape) # smote 전:  (111, 13) (111,)
print("smote 후: ", x_smote_train.shape, y_smote_train.shape) # smote 후:  (159, 13) (159,)
print("smote 전 레이블 값 분포: \n", pd.Series(y_train).value_counts())
# 1    53
# 0    44
# 2    14
print("smote 후 레이블 값 분포: \n", pd.Series(y_smote_train).value_counts())
# 0    53
# 1    53
# 2    53

model2 = XGBClassifier(n_jobs=-1)
model2.fit(x_smote_train, y_smote_train, eval_metric='mlogloss')

score2 = model2.score(x_test, y_test)
print('score : ', score2)
# score :  0.972972972972973 => 성능 향상


