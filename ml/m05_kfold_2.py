import numpy as np
from numpy.core.numeric import cross
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier # tree구조를 앙상블한 형태이다 
import warnings
warnings.filterwarnings('ignore')

'''
KFold, cross_val_score 사용
train test split을 사용하여 train data를 교차검증
'''


datasets = load_iris()
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=9, shuffle=True)


n_splits = 5
kfold = KFold(n_splits=n_splits,  shuffle=True, random_state=66)
# 전체 데이터를 5개로 나누어서 교차검증한다 
# 즉 test data를 전체의 20%로 할당하여 5번 훈련시킨다 

#2.modeling
model = LinearSVC()
# acc [0.85714286 1.         0.95238095 0.95238095 0.95238095]
# 평균  acc:  0.9429

# model = SVC()
# acc [0.9047619  0.95238095 1.         1.         0.9047619 ]
# 평균  acc:  0.9524

# model =  KNeighborsClassifier()
# acc [0.85714286 1.         0.95238095 0.95238095 0.95238095]
# 평균  acc:  0.9429

# model = LogisticRegression()
# acc [0.85714286 1.         1.         1.         0.95238095]
# 평균  acc:  0.9619

# model = DecisionTreeClassifier()
# acc [0.85714286 0.95238095 0.9047619  0.95238095 0.80952381]
# 평균  acc:  0.8952

# model = RandomForestClassifier()
# acc [0.85714286 1.         0.9047619  0.95238095 0.9047619 ]
# 평균  acc:  0.9238

#3. training

#4.predict
scores = cross_val_score(model, x_train, y_train, cv=kfold)
# 한번에 fit, evaluate까지
print('acc',scores)
print('평균  acc: ',round(np.mean(scores), 4))

'''
n_split = 5일 때
[0.96666667 0.96666667 1.         0.9        1.        ]
-> 각각의 교차검증 횟수마다의 acc 
'''

'''
loss:  0.10622021555900574
accuracy:  0.9555555582046509

'''