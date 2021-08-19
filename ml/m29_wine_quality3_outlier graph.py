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
import matplotlib.pyplot as plt

# 다중분류, 0.8이상 완성

'''
./ -> 현재폴더
../ -> 상위폴더
'''

datasets = pd.read_csv('../_data/winequality-white.csv', sep=';',
                        index_col=None, header=0)

print(datasets.shape) # (4898, 12)
print(datasets.info()) 
print(datasets.describe()) 

# 판다스 -> 넘파이로 변환 

# datasets = datasets.to_numpy()
# datasets = datasets.values

# # x와 y를 분리
# x = datasets[:, :11]
# y = datasets[:, 11:]
# # y= np.array(y).reshape(4898, )

# print(x.shape) # (4898, 11)
# print(y.shape) # (4898, 1)


# #sklearn 의 onehot 사용할것
# one_hot_Encoder = OneHotEncoder()
# one_hot_Encoder.fit(y)
# y = one_hot_Encoder.transform(y).toarray()

# print(np.unique(y)) # y라벨 확인

# print(y.shape) # (4898, 7)로 바뀌어야 한다

# x_train, x_test, y_train, y_test = train_test_split(x, y, 
# test_size=0.8, random_state=9, shuffle=True)

# datsets
# x축은 y의 라벨당 개수를 막대 그래프로 그리시오
print(datasets)
'''
      fixed acidity  volatile acidity  ...  alcohol  quality
0               7.0              0.27  ...      8.8        6
1               6.3              0.30  ...      9.5        6
2               8.1              0.28  ...     10.1        6
3               7.2              0.23  ...      9.9        6
4               7.2              0.23  ...      9.9        6
...             ...               ...  ...      ...      ...
4893            6.2              0.21  ...     11.2        6
4894            6.6              0.32  ...      9.6        5
4895            6.5              0.24  ...      9.4        6
4897            6.0              0.21  ...     11.8        6
=> quality가 라벨역할 
'''

count_data = datasets.groupby('quality')['quality'].count()
# quality 컬럼값에 대한 그룹별 개수 count

print(count_data)
# y의 라벨에 대한 count 출력
'''
quality
3      20
4     163
5    1457
6    2198
7     880
8     175
9       5
'''
plt.bar(count_data.index, count_data.values)
plt.show()

# scaler = StandardScaler()
# scaler.fit(x_train)
# scaler.transform(x_train)
# scaler.transform(x_test)


# modeling
# model = XGBClassifier(n_jobs=-1)

# # compile
# model.fit(x_train, y_train)
# # evaluate

# score = model.score(x_test, y_test)
# print('score: ', score)

# y_pred = model.predict(x_test[:5])
# # y_pred = model.predict(x_test[-5:-1])
# print(y_test[:5])
# # print(y_test[-5:-1])
# print('--------softmax를 통과한 값 --------')
# print(y_pred)


# scaler 없을 때
# loss:  2.146963357925415
# accuracy:  0.47389909625053406

# minMax Scaler
# loss:  2.8873708248138428
# accuracy:  0.4234470725059509

# QuantileTransfomer
# loss:  1.7815757989883423
# accuracy:  0.4849810302257538

# PowerTransformer
# accuracy:  0.5191017985343933

# StandardScaler
# loss:  1.235440969467163
# accuracy:  0.5281423330307007

# model = XGBClassifier(n_jobs=-1)
# score:  0.5815106445027705

