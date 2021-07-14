from sklearn.datasets import load_boston # 교육용 예제 임포트
from tensorflow.python.keras.models import Sequential 
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# boston 주택 가격 예측

#1. data
datasets = load_boston()
x = datasets.data
y = datasets.target

print(datasets.feature_names)

print(np.min(x), np.max(x)) # 0.0 711.0
print(np.min(y), np.max(y)) # 5.0 50.0


# 데이터 전처리
# x = x/np.max(x)
# x = (x-np.min(x))/(np.max(x)-np.min(x))
# x의 값만 정규화 시켜주고 y는 변화시키지 않는다. 

# 함수를 만들어서 전처리 

def processing():
    for i in range(0, len(x)):
        x[i] = (x[i]-np.min(x))/(np.max(x)-np.min(x))
        np.append(x, x[i])
    return print(x)
processing()


# 기존의 MinMaxScaler()를 이용하여 전처리 
scaler = MinMaxScaler()
scaler.fit(x) # 훈련 -> 메모리 상태에서 실행
x_scale = scaler.transform(x) # transform = 변환한 상태를 배치시키는 것 
# 위에 x 대신 x_test를 넣어주게 되면 

print(x_scale[:10])
print(np.min(x_scale), np.max(x_scale))


x_train, x_test, y_train, y_test = train_test_split(x_scale, y, 
train_size=0.7, shuffle=True, random_state=9)

print(x.shape) # (506, 13)
print(x_train.shape) 
# (354, 13) -> 전체 데이터가 아닌 일부 데이터로 학습해야  성능이 좋아진다
print(x_test.shape) # (152, 13)


# 2.model
model = Sequential()
model.add(Dense(400, input_dim=13, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

# 3. compile
model.compile(loss='mse', optimizer="adam")

model.fit(x_train, y_train, epochs=100, batch_size=1)
# model.fit(x_train, y_train, epochs=100) -> batch_size는 default로 들어가 있다
# 값은 32가 기본값


# 4. evaluate and predict
loss = model.evaluate(x_test, y_test)
print('loss:', loss)

# loss: 37.513221740722656
# loss: 18.4830265045166
# loss: 16.495746612548828


y_pred = model.predict(x_test)
print('x_test를 통한 y의 예측값:', y_pred)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred) # y_test, y_pred 차이
print('r2스코어:',r2)

# r2스코어: 0.3354429571900964
# r2스코어: 0.7917688093385801
# r2스코어: 0.8269183397027154

# # 시각화 
# plt.scatter(y_test, y_pred)
# plt.plot(x, y_pred, color='red')
# # x와 x_test를 통해 예측한 y의 값을 그래프로 나타낸다
# plt.show()


# minmax 전처리 이후
# loss: 15.998124122619629 -> 정규화 이후
# r2스코어: 0.8197639152418532 -> 정규화 이후


# MinMaxScaler 정규화 이후
# loss: 8.033093452453613
# r2스코어: 0.9094985498559746


