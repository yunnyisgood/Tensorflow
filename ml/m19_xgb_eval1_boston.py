from xgboost import XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer
import matplotlib.pyplot as plt

'''
feature가 아닌 model engineering

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
model = XGBRegressor(n_estimators=20, learning_rate = 0.05, n_jobs=1)

# 3. fit
model.fit(x_train, y_train, verbose=1, eval_metric=['rmse', 'mae', 'logloss'],
          eval_set=[(x_train, y_train), (x_test, y_test)]
)          
                        #  훈련 set,     validation set

# 4.
result = model.score(x_test, y_test)
print('result: ', result)

y_pred = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred) # y_test, y_pred 차이

print('r2스코어:',r2)

print("======================================")

results = model.evals_result()
epochs = len(results['validation_0']['rmse'])
x_axis = range(0, epochs)
fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['rmse'], label='rmse_Train')
ax.plot(x_axis, results['validation_1']['rmse'], label='rmes_Test')
ax.plot(x_axis, results['validation_0']['mae'], label='mae_Train')
ax.plot(x_axis, results['validation_1']['mae'], label='mae_Test')
ax.legend()
plt.ylabel('rmse')
plt.title('XGBoost rmse')
plt.show()

'''
선생님 pyplot

epochs = len(results['validation_0']['logloss'])
x_axis = range(0, epochs)
fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
ax.legend()
plt.ylabel('Log Loss')
plt.title('XGBoost Log Loss')
# plt.show()

'''

'''
default
result:  0.865766574050506
r2스코어: 0.865766574050506

n_estimators=10000
result:  0.8657622772286538
r2스코어: 0.8657622772286538
'''