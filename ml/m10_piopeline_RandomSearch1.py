import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.utils import all_estimators
from sklearn.metrics import r2_score
import time
from sklearn.pipeline import make_pipeline, Pipeline
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

datasets = load_iris()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, 
train_size=0.7, shuffle=True, random_state=9)

n_splits = 5
kfold = KFold(n_splits=n_splits,  shuffle=True, random_state=66)


# parameters = [
#     {'n_estimators':[100, 200], 'max_depth':[6, 8, 10], 'min_samples_leaf':[5, 7, 10]}, 
#     {'max_depth':[5, 6, 7], 'min_samples_leaf':[3, 6, 9, 11], 'min_samples_split':[3, 4, 5]},
#     {'min_samples_leaf':[3, 5, 7], 'min_samples_split':[2, 3, 5, 10]},
#     { 'min_samples_split':[2, 3, 5, 10]}
# ]

# parameters = [
#     {'rf__min_samples_leaf':[3, 5, 7], 'rf__max_depth':[2, 3, 5, 10]},
#     { 'rf__min_samples_split':[6, 8, 10]}
# ]
# 2)
# 아래 오류의 해결 방법! -> 모델명을 소문자로 작성해서 파라미터 앞에 작성해준다 

parameters = [

    {"xgb__n_estimators":[100, 200, 300], "xgb__learning_rate": [0.1, 0.3, 0.001, 0.01],
    "xgb__max_depth": [4,5,6]},
    {"xgb__n_estimators":[90, 100, 110], "xgb__learning_rate": [0.1,  0.001, 0.01],
    "xgb__max_depth": [4,5,6], "xgb__colsample_bytree":[0.6, 0.9, 1]},
     {"xgb__n_estimators":[90, 110], "xgb__learning_rate": [0.1,  0.001, 0.5],
    "xgb__max_depth": [4,5,6], "xgb__colsample_bytree":[0.6, 0.9, 1],
    "xgb__colsample_bylevel": [0.6, 0.7, 0.9]}

]
n_jobs = -1

pipe = Pipeline([("scaler", MinMaxScaler()), ("xgb", XGBClassifier())])
# alias를 사용할 수 있다 

model = RandomizedSearchCV(pipe, parameters, cv=kfold, verbose=1)

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


'''
>> model = GridSearchCV(RandomForestRegressor(), parameters, cv=kfold)

최적의 매개변수:  RandomForestRegressor(min_samples_leaf=10)
best_score_:  0.4068137866834106
model.score:  0.5212257760438228
r2 score:  0.5212257760438228

parametr 변경 후
최적의 매개변수:  RandomForestRegressor(max_depth=7, min_samples_leaf=11, min_samples_split=5,
                      n_jobs=-1)
best_score_:  0.4186712411050243
model.score:  0.5135251447727294
r2 score:  0.5135251447727294
걸린시간 :  44.93169665336609

n_jobs =-1 없앴을 때
최적의 매개변수:  RandomForestRegressor(max_depth=7, min_samples_leaf=11, min_samples_split=3)
best_params:  {'max_depth': 7, 'min_samples_leaf': 11, 'min_samples_split': 3}
best_score_:  0.41984890548270626
model.score:  0.510046148137179
r2 score:  0.510046148137179
걸린시간 :  42.204020738601685

RandomizedSearch로 변경
적의 매개변수:  RandomForestRegressor(max_depth=6, min_samples_leaf=10, n_estimators=200)
best_params:  {'n_estimators': 200, 'min_samples_leaf': 10, 'max_depth': 6}
best_score_:  0.41609882939513126
model.score:  0.5217402565488081
r2 score:  0.5217402565488081
걸린시간 :  6.5225701332092285

make_pipeline, GridSearchCV
최적의 매개변수:  Pipeline(steps=[('minmaxscaler', MinMaxScaler()),
                ('randomforestclassifier',
                 RandomForestClassifier(max_depth=3, min_samples_leaf=7))])
best_params:  {'randomforestclassifier__max_depth': 3, 'randomforestclassifier__min_samples_leaf': 7}
best_score_:  0.9333333333333333
model.score:  1.0
정답률:  1.0


make_pipeline, RandomizedSearchCV
최적의 매개변수:  Pipeline(steps=[('minmaxscaler', MinMaxScaler()),
                ('randomforestclassifier',
                 RandomForestClassifier(min_samples_split=10))])
best_params:  {'randomforestclassifier__min_samples_split': 10}
best_score_:  0.9333333333333333
model.score:  1.0
정답률:  1.0

Pipeline, GridSearchCV
최적의 매개변수:  Pipeline(steps=[('scaler', MinMaxScaler()),
                ('rf',
                 RandomForestClassifier(max_depth=10, min_samples_leaf=3))])
best_params:  {'rf__max_depth': 10, 'rf__min_samples_leaf': 3}
best_score_:  0.9238095238095239
model.score:  1.0
정답률:  1.0

Pipeline, RandomizedSearchCV
최적의 매개변수:  Pipeline(steps=[('scaler', MinMaxScaler()),
                ('rf',
                 RandomForestClassifier(max_depth=5, min_samples_leaf=7))])
best_params:  {'rf__min_samples_leaf': 7, 'rf__max_depth': 5}
best_score_:  0.9238095238095239
model.score:  1.0
정답률:  1.0
'''