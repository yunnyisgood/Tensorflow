from xgboost import XGBRegressor
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer
from xgboost.sklearn import XGBClassifier

'''
분류 -> eval_metric을 찾아서 추가

'''

# 1. data
dataset = load_iris()
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
model = XGBClassifier(n_estimators=300, learning_rate = 0.1, n_jobs=1)
# n_estimators=100, learning_rate = 0.05, n_jobs=1

# 3. fit
model.fit(x_train, y_train, verbose=1, eval_metric=['logloss', 'merror'],
          eval_set=[(x_train, y_train), (x_test, y_test)]
)          
                        #  훈련 set,     validation set

# 4.
result = model.score(x_test, y_test)
print('result: ', result)

# y_pred = model.predict(x_test)
# from sklearn.metrics import r2_score
# r2 = r2_score(y_test, y_pred) # y_test, y_pred 차이

# print('r2스코어:',r2)

'''
default
result:  0.9

n_estimators=100, learning_rate = 0.05, n_jobs=1
result:  0.9333333333333333
'''