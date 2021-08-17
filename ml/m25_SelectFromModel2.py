from inspect import Parameter
from math import nan
from sklearn import datasets
from xgboost import XGBRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings
import time
warnings.filterwarnings('ignore')

'''
실습
1. GridSerarch, Randomserch로 모델 구성
최적의 r2, feature_importnaces

2. select_model

3. 갯수 조정해서 다시 최적의 r2

4. 0.47이상으로 끌어올리기 
'''

x, y = load_diabetes(return_X_y=True)
print(x.shape, y.shape)
# (506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(x, y, 
train_size=0.7, shuffle=True, random_state=9)

n_splits = 5
kfold = KFold(n_splits=n_splits,  shuffle=True, random_state=66)

# parameters = [
#     {'n_estimators':[100, 200], 'max_depth':[6, 8, 10], 'min_samples_leaf':[5, 7, 10], 'min_samples_split':[2, 3, 5, 10]}
# ]
# # 'n_estimators':[100, 200] -> epochs와 동일한 역할. 몇번 훈련시킬지에 대한 parameters

# parameters = [{'n_estimators': [200], 'min_samples_leaf': [10], 'max_depth': [6]}]

# #2.modeling
# model = RandomizedSearchCV(RandomForestRegressor(), parameters,verbose=1)
# model = GridSearchCV(XGBRegressor(), parameters,verbose=1)

# model = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
#              importance_type='gain', interaction_constraints='',
#              learning_rate=0.300000012, max_delta_step=0, max_depth=6,
#              min_child_weight=1, min_samples_leaf=5, min_samples_split=2,
#              missing=nan, monotone_constraints='()', n_estimators=100, n_jobs=8,
#              num_parallel_tree=1, random_state=0, reg_alpha=0, reg_lambda=1,
#              scale_pos_weight=1, subsample=1, tree_method='exact',
#              validate_parameters=1, verbosity=None)

model = RandomForestRegressor(max_depth=6, min_samples_leaf=10, n_estimators=200)

#3. training
start_time = time.time()
model.fit(x_train, y_train)

#4.predict
# 4-1 : train값으로 훈련을 했을 때 정확도
# print("최적의 매개변수: ", model.best_estimator_)
# print("best_score_: ", model.best_score_)

# 4-2 : test값을 따로 빼서 훈련을 거치지 않은 값들로 학습을 시킨 뒤 평가했을 때 
print("model.score: ", model.score(x_test, y_test))

end_time = time.time() - start_time
print("걸린시간 : ", end_time)

threshhold = np.sort(model.feature_importances_)
print(threshhold)

for thresh in threshhold:
    print(thresh)
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    # print('selection: ', selection)

    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    print(select_x_train.shape, select_x_test.shape)

    selection_model = RandomForestRegressor(max_depth=6, min_samples_leaf=10, n_estimators=200)
    selection_model.fit(select_x_train, y_train)

    y_pred = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_pred)

    print("Thresh=%.3f, n=%d, R2: %.2f%%" %(thresh, select_x_train.shape[1],
            score*100))



'''
model = XGBRegressor(n_jobs=8)
model.score:  0.23802704693460175


model.score:  0.44221928509211916
걸린시간 :  0.08776450157165527
[0.02281273 0.04124952 0.04668248 0.04762195 0.05755796 0.05904608
 0.07138201 0.07221497 0.11473364 0.46669868]
0.022812733
(309, 10) (133, 10)
Thresh=0.023, n=10, R2: 44.22%
0.041249517
(309, 9) (133, 9)
Thresh=0.041, n=9, R2: 41.06%
0.046682484
(309, 8) (133, 8)
Thresh=0.047, n=8, R2: 41.91%
0.04762195
(309, 7) (133, 7)
Thresh=0.048, n=7, R2: 39.74%
0.057557955
(309, 6) (133, 6)
Thresh=0.058, n=6, R2: 37.52%
0.05904608
(309, 5) (133, 5)
Thresh=0.059, n=5, R2: 34.36%
0.07138201
(309, 4) (133, 4)
Thresh=0.071, n=4, R2: 35.20%
0.07221497
(309, 3) (133, 3)
Thresh=0.072, n=3, R2: 20.83%
0.11473364
(309, 2) (133, 2)
Thresh=0.115, n=2, R2: 15.80%
0.46669868
(309, 1) (133, 1)
Thresh=0.467, n=1, R2: -17.52%


model = RandomForestRegressor(max_depth=6, min_samples_leaf=10, n_estimators=200)
model.score:  0.5324986995716755
걸린시간 :  0.22143936157226562
[0.01037557 0.01492696 0.01601359 0.01792684 0.01874291 0.03237307
 0.04174118 0.12224727 0.21518197 0.51047065]
0.0103755653130059
(309, 10) (133, 10)
Thresh=0.010, n=10, R2: 52.05%
0.014926955240318831
(309, 9) (133, 9)
Thresh=0.015, n=9, R2: 51.32%
0.016013587978562976
(309, 8) (133, 8)
Thresh=0.016, n=8, R2: 52.27% >>
0.01792683705230219
(309, 7) (133, 7)
Thresh=0.018, n=7, R2: 51.27%
0.01874290966655738
(309, 6) (133, 6)
Thresh=0.019, n=6, R2: 50.87%
0.032373067982862345
(309, 5) (133, 5)
Thresh=0.032, n=5, R2: 49.90%
0.041741184676444915
(309, 4) (133, 4)
Thresh=0.042, n=4, R2: 48.06%
0.12224727218901166
(309, 3) (133, 3)
Thresh=0.122, n=3, R2: 48.36%
0.215181966994459
(309, 2) (133, 2)
Thresh=0.215, n=2, R2: 49.03%
0.5104706529064748
(309, 1) (133, 1)
Thresh=0.510, n=1, R2: 24.22%


'''

