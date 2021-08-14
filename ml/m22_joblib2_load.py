from xgboost import XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer
import matplotlib.pyplot as plt
import pickle

'''
joblib 사용

'''

# 1. data
dataset = load_boston()
x = dataset['data']
y = dataset['target']

print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, 
train_size=0.8, shuffle=True, random_state=66)

scaler = PowerTransformer()
scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)

# 2. modeling
# model = XGBRegressor(n_estimators=2000, learning_rate = 0.05, n_jobs=1)

# # 3. fit
# model.fit(x_train, y_train, verbose=1, eval_metric=['rmse', 'mae', 'logloss'],
#           eval_set=[(x_train, y_train), (x_test, y_test)],
#           early_stopping_rounds=10
# )          
                        #  훈련 set,     validation set

# 4.
# result = model.score(x_test, y_test)
# print('result: ', result)

# y_pred = model.predict(x_test)
# from sklearn.metrics import r2_score
# r2 = r2_score(y_test, y_pred) # y_test, y_pred 차이

# print('r2스코어:',r2)

# print("======================================")

# evals_result = model.evals_result()
# print(evals_result)
# # # hist = model.fit에서의 history와 동일한 역할 
# # 훈련횟수에 따라 어떻게 변화하는지 보여준다

# # n_estimator, evals_result 그래프


# 저장
# pickle.dump(model, open('./_save/xgb_save/m21.picke.dat', 'wb'))

import joblib
# joblib.dump(model, './_save/xgb_save/m22.joblib.dat')

model = joblib.load('./_save/xgb_save/m22.joblib.dat')

result = model.score(x_test, y_test)
print('result: ', result)

y_pred = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred) # y_test, y_pred 차이

print('r2스코어:',r2)

print("======================================")

evals_result = model.evals_result()
print(evals_result)