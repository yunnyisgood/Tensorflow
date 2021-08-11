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
from sklearn.model_selection import KFold, cross_val_score


#1. data
datasets = load_boston()
x = datasets.data
y = datasets.target


n_splits = 5
kfold = KFold(n_splits=n_splits,  shuffle=True, random_state=66)

#2.modeling
# model = KNeighborsRegressor()
# r2스코어: 0.716098217736928

# model = LinearRegression()
# r2스코어: 0.7406426641094095

# model =  DecisionTreeRegressor()
# r2스코어: 1.0

model = RandomForestRegressor()
# r2스코어: 0.9828969681967524

model.fit(x, y)

scores = cross_val_score(model, x, y, cv=kfold)
# 한번에 fit, evaluate까지
print('acc',scores)
print('평균  acc: ',round(np.mean(scores), 4))


y_pred = model.predict(x)


from sklearn.metrics import r2_score
r2 = r2_score(y, y_pred) # y_test, y_pred 차이

print('r2스코어:',r2)

