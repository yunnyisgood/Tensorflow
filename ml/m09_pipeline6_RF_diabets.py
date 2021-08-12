import numpy as np
from numpy.core.numeric import cross
from sklearn.datasets import load_diabetes
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
from sklearn.ensemble import RandomForestRegressor # tree구조를 앙상블한 형태이다 
import warnings
from sklearn.pipeline import make_pipeline, Pipeline

'''
pipeline을 사용하여 scaling
'''

warnings.filterwarnings('ignore')

datasets = load_diabetes()
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, 
train_size=0.7, shuffle=True, random_state=9)

#2.modeling
model = make_pipeline(MinMaxScaler(), RandomForestRegressor())
# pipeline을 사용하여 scaling, modeling을 한번에

#3. training
model.fit(x_train, y_train)

#4.predict
# 4-1 : train값으로 훈련을 했을 때 정확도
# print("최적의 매개변수: ", model.best_estimator_)
# print("best_score_: ", model.best_score_)

# 4-2 : test값을 따로 빼서 훈련을 거치지 않은 값들로 학습을 시킨 뒤 평가했을 때 
print("model.score: ", model.score(x_test, y_test))

y_pred  = model.predict(x_test)

print("r2 score: ", r2_score(y_test, y_pred))

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
최적의 매개변수:  RandomForestRegressor(max_depth=5, min_samples_leaf=11, min_samples_split=5)
best_score_:  0.42033001733757536
model.score:  0.5234244776338356
r2 score:  0.5234244776338356
걸린시간 :  41.48382568359375

make_pipeline, RandomClassfier
model.score:  0.49747683433884804
'''



'''
loss:  0.10622021555900574
accuracy:  0.9555555582046509

'''