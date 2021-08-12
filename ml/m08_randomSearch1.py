import numpy as np
from numpy.core.numeric import cross
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
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
GridSearchCV 사용
-> 체로 걸러낸 cross validation data를 찾겠다 라는 의미

단젇으로는 교차검증과 같이 시간이 오래 걸린다

'''


datasets = load_iris()
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, 
train_size=0.7, shuffle=True, random_state=9)

n_splits = 5
kfold = KFold(n_splits=n_splits,  shuffle=True, random_state=66)

# print(SVC.get_params().keys())

parameters = [
    {"C":[1, 10, 100, 1000], "kernel":["linear"]}, # 1, 10, 100, 1000 총 4개 X n_plits(=5) = 20 번 돌아감 
    {"C":[1, 10, 100], "kernel":["rbf"],"gamma":[0.001, 0.0001]}, # 3 X 1 X 2 X 5 = 30번
    {"C":[1, 10, 100, 1000], "kernel":["sigmoid"], "gamma":[0.001, 0.0001]} # 4 X 1 X 2 X 5 = 90번
]

#2.modeling
# model = GridSearchCV(SVC(), parameters, cv=kfold, verbose=1)
# Fitting 5 folds for each of 18 candidates, totalling 90 fits

model = RandomizedSearchCV(SVC(), parameters, cv=kfold, verbose=1)
# Fitting 5 folds for each of 10 candidates, totalling 50 fits

#3. training
model.fit(x_train, y_train)

#4.predict
# 4-1 : train값으로 훈련을 했을 때 정확도
print("최적의 매개변수: ", model.best_estimator_)
print("best_score_: ", model.best_score_)

# 4-2 : test값을 따로 빼서 훈련을 거치지 않은 값들로 학습을 시킨 뒤 평가했을 때 
print("model.score: ", model.score(x_test, y_test))

y_pred  = model.predict(x_test)
print("정답률: ", accuracy_score(y_test, y_pred))

'''
> GridSearch
최적의 매개변수:  SVC(C=1, kernel='linear')
best_score_:  0.9619047619047618
model.score:  1.0
정답률:  1.0

> RandomizedSearch
최적의 매개변수:  SVC(C=1000, kernel='linear')
best_score_:  0.9619047619047618
model.score:  1.0
정답률:  1.0
'''



'''
loss:  0.10622021555900574
accuracy:  0.9555555582046509

'''