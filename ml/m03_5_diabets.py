import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
from sklearn.svm import LinearSVC, SVC # 되는지 확인 -> 안되면 아래 regrsseion 모델로 변경해서 다시 실습 
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


# 1. 데이터
datasets  = load_diabetes()
x = datasets.data
y = datasets.target

print(x.shape) # (442, 10)
print(y.shape) # (442, )

print(datasets.feature_names)
# 'age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
# 10가지의 지수를 통해 당뇨병 지수를 파악한다 

print(datasets.DESCR)

x_train, x_test, y_train, y_test = train_test_split(x, y, 
train_size=0.8, shuffle=True, random_state=9)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler =  RobustScaler()
scaler =  PowerTransformer()
scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)


#2.modeling
# model = KNeighborsRegressor()
# model.score:  0.48322394424037063

# model = LinearRegression()
#model.score:  0.5851141269959736

# model =  DecisionTreeRegressor()
# model.score:  -0.33803254507122715

model = RandomForestRegressor()
# model.score:  0.5433808986485881


#3. training
model.fit(x_train, y_train)

#4.predict
y_pred = model.predict(x_test)
y_pred = np.rint(y_pred) # ->  반올림을 통해 0 아니면 1의 값으로 y_pred를 변형 
print(x_train,'의 예측결과 : ', y_pred)

result = model.score(x_test, y_test)
print('model.score: ', result)

# acc = accuracy_score(y_test, y_pred)
# print('accuracy_score: ', acc)

'''
QuantileTransformer, MinMaxScaler 순서로 성능 우수

'''

# MinMaxScaler 전처리 이후
# loss: 2097.257080078125
# r2 스코어:  0.6139346339266243

# StandardScaler 전처리 이후
# loss: 2163.699462890625
# r2 스코어:  0.6024054583045837

# MaxAbsScaler 전처리 이후
# loss: 2173.5859375
# r2 스코어:  0.6005887828234024

# RobustScaler 전처리 이후
# loss: 2341.55224609375
# r2 스코어:  0.56972377635531

# QuantileTransformer 전처리 이후
# loss: 2079.048828125
# r2 스코어:  0.6179605805519554

# PowerTransformer 전처리 이후
# loss: 2284.801025390625
# r2 스코어:  0.5801522006661404
