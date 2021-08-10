import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
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
from sklearn.ensemble import RandomForestClassifier # tree구조를 앙상블한 형태이다 
# sklearn의 모델에는 classfier, regression 2개 


# 다중분류, One-Hot-Encoding 

datasets = load_iris()
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=9, shuffle=True)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# (105, 4) (45, 4) (105,) (45,)

scaler = MinMaxScaler()
scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)

#2.modeling
# model = LinearSVC()
# model.score:  1.0
# accuracy_score:  1.0

# model = SVC()
# model.score:  0.9777777777777777
# accuracy_score:  0.9777777777777777

# model =  KNeighborsClassifier()
# model.score:  1.0
# accuracy_score:  1.0

# model = LogisticRegression()
# model.score:  1.0
# accuracy_score:  1.0

# model = DecisionTreeClassifier()
# model.score:  0.9777777777777777
# accuracy_score:  0.9777777777777777

model = RandomForestClassifier()
# model.score:  1.0
# accuracy_score:  1.0

#3. training
model.fit(x_train, y_train)

#4.predict
y_pred = model.predict(x_test)
y_pred = np.rint(y_pred) # ->  반올림을 통해 0 아니면 1의 값으로 y_pred를 변형 
print(x_train,'의 예측결과 : ', y_pred)

result = model.score(x_test, y_test)
print('model.score: ', result)

acc = accuracy_score(y_test, y_pred)
print('accuracy_score: ', acc)

'''
loss:  0.10622021555900574
accuracy:  0.9555555582046509

'''