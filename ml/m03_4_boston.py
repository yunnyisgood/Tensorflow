'''
회귀 데이터를 classfier로 만들었을 때 에러 확인 

'''
from sklearn.svm import LinearSVC, SVC # 되는지 확인 -> 안되면 아래 regrsseion 모델로 변경해서 다시 실습 
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer, StandardScaler, MinMaxScaler
from sklearn.datasets import load_boston 
from tensorflow.python.keras.models import Sequential 
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score



#1. data
datasets = load_boston()
x = datasets.data
y = datasets.target

print(x.shape) # (506, 13) -> 13가지의 특성 -> Input_dim = 13
print(y.shape) # (506,) -> output_dim = 1

print(datasets.feature_names)

x_train, x_test, y_train, y_test = train_test_split(x, y, 
train_size=0.7, shuffle=True, random_state=9)

scaler = PowerTransformer()
scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)


#2.modeling
# model = KNeighborsRegressor()
# model.score:  0.5941848521354716

# model = LinearRegression()
# model.score:  0.7826126074271011

# model =  DecisionTreeRegressor()
# model.score:  0.7180730967098297

model = RandomForestRegressor()
# model.score:  0.8378618739886178


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
classfier로 모델링 했을 때 오류:
ValueError: Unknown label type: 'continuous'

accuracy_score 오류:
ValueError: Classification metrics can't handle a mix of continuous and multiclass targets


'''

'''
PowerTransformer, MinMaxScaler, StandardScaler 순서로
가장 성능이 우수
 
'''

# r2스코어: 0.3354429571900964
# r2스코어: 0.7853412907684698

# MinMaxScaler 전처리 이후
# loss: 17.463186264038086
# r2스코어: 0.8032583968614756

# StandardScaler 전처리 이후
# loss: 18.29416847229004
# r2스코어: 0.793896502480951

# MaxAbsScaler 전처리 이후
# loss: 25.627376556396484
# r2스코어: 0.7112800084011388

# RobustScaler 전처리 이후
# loss: 22.079360961914062
# r2스코어: 0.751252205269121

# QuantileTransformer 전처리 이후
# loss: 25.287765502929688
# r2스코어: 0.7151061161907877

# PowerTransformer 전처리 이후
# loss: 17.08149528503418
# r2스코어: 0.807558585442023