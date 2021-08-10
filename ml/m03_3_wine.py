import numpy as np
from sklearn.datasets import load_wine
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, PowerTransformer, QuantileTransformer
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

'''
완성
accuracy 0.8 이상 만들어 볼 것
=> 0.9279999732971191 가 현재까지 최대
'''

datasets = load_wine()

print(type(datasets))  # <class 'sklearn.utils.Bunch'>
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target

print(x.shape) # (178, 13)
print(y.shape) # (178, )

# y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, 
test_size=0.7, random_state=9, shuffle=True)

scaler = MinMaxScaler()
scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)

#2.modeling
# model = LinearSVC()
# model.score:  0.688
# accuracy_score:  0.688

# model = SVC()
# model.score:  0.672
# accuracy_score:  0.672

# model =  KNeighborsClassifier()
# model.score:  0.656
# accuracy_score:  0.656

# model = LogisticRegression()
# model.score:  0.936
# accuracy_score:  0.936

# model = DecisionTreeClassifier()
# model.score:  0.936
# accuracy_score:  0.936

model = RandomForestClassifier()
# model.score:  0.976
# accuracy_score:  0.976

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
# 범위 -> [:5]

# scaler 없이
# loss:  0.750230073928833
# accuracy:  0.671999990940094


# MinMax scaler 사용할 때
# loss:  0.5610019564628601
# accuracy:  0.7839999794960022

# -> MinMax Scaler인데 node 조정 후
# loss:  0.24176639318466187
# accuracy:  0.9279999732971191

# PowerTransformer
# loss:  0.8162326812744141
# accuracy:  0.6320000290870667

# QuantileTransformer
# loss:  0.6085973381996155
# accuracy:  0.6800000071525574

# 범위 -> [-5: -1]
# loss:  0.6953398585319519
# accuracy:  0.6320000290870667