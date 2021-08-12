import numpy as np
from sklearn.datasets import load_wine
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
from sklearn.ensemble import RandomForestClassifier 
from sklearn.utils import all_estimators
import time
import warnings
warnings.filterwarnings('ignore')

datasets = load_wine()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, 
train_size=0.7, shuffle=True, random_state=9)

n_splits = 5
kfold = KFold(n_splits=n_splits,  shuffle=True, random_state=66)

# parameters = [

#     {'n_estimators':[100, 200]},
#     {'max_depth': [6, 8, 10, 12]},
#     {'min_samples_leaf': [3, 5, 7, 10]},
#     {'min_samples_split': [2, 3, 5, 10]},
#     {'n_jobs': [-1, 2, 4]}

# ]

parameters = [
    {'n_estimators':[100, 200]},
    {'max_depth':[6, 8, 10, 12]},
    {'min_samples_leaf':[3, 5, 7, 10]},
    {'min_samples_split':[2, 3, 5, 10]},
    {'n_jobs':[-1, 2, 4]}
]


#2.modeling
# model = GridSearchCV(RandomForestClassifier(), parameters, cv=kfold, verbose=1)
# Fitting 5 folds for each of 17 candidates, totalling 85 fits

model = RandomizedSearchCV(RandomForestClassifier(), parameters, cv=kfold, verbose=1)
# Fitting 5 folds for each of 10 candidates, totalling 50 fits

#3. training
start_time = time.time()
model.fit(x_train, y_train)

#4.predict
# 4-1 : train값으로 훈련을 했을 때 정확도
print("최적의 매개변수: ", model.best_estimator_)
print("best_params: ", model.best_params_)
print("best_score_: ", model.best_score_)

# 4-2 : test값을 따로 빼서 훈련을 거치지 않은 값들로 학습을 시킨 뒤 평가했을 때 
print("model.score: ", model.score(x_test, y_test))

y_pred  = model.predict(x_test)
print("정답률: ", accuracy_score(y_test, y_pred))

print("걸린 시간 : ", time.time()-start_time)

'''

>> model = GridSearchCV(RandomForestClassifier(), parameters, cv=kfold)

최적의 매개변수:  RandomForestClassifier(max_depth=10)
best_params:  {'max_depth': 10} 
best_score_:  0.984
model.score:  0.9814814814814815
정답률:  0.9814814814814815     
걸린 시간 :  12.465778827667236 

RandomizedSearch
최적의 매개변수:  RandomForestClassifier(min_samples_leaf=3)
best_params:  {'min_samples_leaf': 3}
best_score_:  0.984
model.score:  0.9814814814814815
정답률:  0.9814814814814815   
걸린 시간 :  6.382002115249634

'''