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
from sklearn.metrics import accuracy_score, f1_score
# f1_score는 이진분류에서 사용

warnings.filterwarnings('ignore')


datasets = pd.read_csv('../_data/winequality-white.csv', sep=';',
                        index_col=None, header=0)

print(datasets.shape) # (4898, 12)
print(datasets.info()) 
print(datasets.describe()) 

# 판다스 -> 넘파이로 변환 

datasets = datasets.to_numpy()

# x와 y를 분리
x = datasets[:, :11]
y = datasets[:, 11:]
y= np.array(y).reshape(4898, )

print(x.shape, y.shape) # (4898, 11) (4898, 1)

print(pd.Series(y).value_counts())
# 6.0    2198
# 5.0    1457
# 7.0     880
# 8.0     175
# 4.0     163
# 3.0      20
# 9.0       5

# label 축소
# 위의 9.0의 값이 5개밖에 되지 않기 때문에 8.0에 통합
# 방법 1

# y = np.where(y==9.0, 7.0, y)
# y = np.where(y==3.0, 5.0, y)
# y = np.where(y==8.0, 7.0, y)
# y = np.where(y==4.0, 5.0, y)

y = np.where(y==9.0, 2, y)
y = np.where(y==8.0, 2, y)
y = np.where(y==7.0, 1, y)
y = np.where(y==6.0, 1, y)
y = np.where(y==5.0, 1, y)
y = np.where(y==4.0, 0, y)
y = np.where(y==3.0, 0, y)




# 방법 2
# for index, value in enumerate(y):
#     if value ==9:
#         y[index] = 8        
        
print(pd.Series(y).value_counts())
# 6.0    2198
# 5.0    1457
# 7.0     880
# 8.0     180
# 4.0     163
# 3.0      20

# 이렇게 데이터 전체가 아니라 부분으로 슬라이싱 한 뒤, 추가로 smote를 사용해 데이터 증폭했을때
# 성능이 잘 나옴을 알 수 있다. 
# x = x[:-400]
# y = y[:-400]

print(x.shape, y.shape)

# x_new = x[:-200]
# y_new = y[:-200]

# print(x_new.shape, y_new.shape)

# x_train,x_test, y_train, y_test =  train_test_split(
#     x_new, y_new, train_size=0.8, shuffle=True, random_state=66, 
#     stratify=y_new # trian, test size의 비율에 맞게 라벨값이 분리되어 진다 
# )

x_train,x_test, y_train, y_test =  train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=66, 
    stratify=y # trian, test size의 비율에 맞게 라벨값이 분리되어 진다 
)
# train_size = 0.75가 기본값

print(pd.Series(y_train).value_counts())
# 6.0    1648
# 5.0    1093
# 7.0     660
# 8.0     131
# 4.0     122
# 3.0      15
# 9.0       4

model = XGBClassifier(n_jobs=-1)

model.fit(x_train, y_train, eval_metric='mlogloss')

score = model.score(x_test, y_test)
print('score : ', score)
# score :  0.643265306122449

y_pred = model.predict(x_test)
f1 = f1_score(y_test, y_pred, average='macro')
print('f1_score: ', f1)

################### SMOTE 적용 #####################
print("===============SMOTE적용================")

smote = SMOTE(random_state=66, k_neighbors=60) # , k_neighbors=60) -> 가지가 많아지는 것. 즉 연산이 많아지기 때문에
# 성능은 좋아질 수 있지만 시간이 걸릴 수 있다
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
# score :  0.6318367346938776 => 성능 떨어짐'''

y_pred = model2.predict(x_test)
f1 = f1_score(y_test, y_pred, average='macro')
print('f1_score: ', f1)

'''
ValueError: Expected n_neighbors <= n_samples,  but n_samples = 4, n_neighbors = 6
=> 라벨당 값의 개수가 최소 6개가 존재되어야 smote를 사용할 수 있다
=> 
해결방법 1)
smote = SMOTE(random_state=66, k_neighbors=3)
위에서 k_neighbors 파라미터를 통해 최소 n_neighbors를 지정할 수 있다
n_neighbors default값은 5
-> 3으로 조정할 수 있으나, 값을 내리면 score는 떨어짐. 

해결방법 2)
라벨 축소
score :  0.6236734693877551 => 성능 더 떨어짐
=>전체 데이터를 사용하는 것이 아니라 부분적으로 데이터를 슬라이싱 한 뒤
smote를 사용해서 증폭해야지 성능이 개선된다. 


=> model2.score의 성능이 더 좋게 나오도록 개선해야

'''


