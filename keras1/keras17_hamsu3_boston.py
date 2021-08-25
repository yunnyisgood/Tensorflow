from sklearn.datasets import load_boston # 교육용 예제 임포트
from tensorflow.python.keras.models import Sequential, Model 
from tensorflow.keras.layers import Dense, Input
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# boston 주택 가격 예측

#1. data
datasets = load_boston()
x = datasets.data
y = datasets.target

print(x.shape) # (506, 13) -> 13가지의 특성 -> Input_dim = 13
print(y.shape) # (506,) -> output_dim = 1
# print(datasets.DESCR)

print(datasets.feature_names)

x_train, x_test, y_train, y_test = train_test_split(x, y, 
train_size=0.7, shuffle=True, random_state=66)

# 2.model
input1 = Input(shape=(13, ))
dense1 = Dense(300)(input1)
dense2 = Dense(100)(dense1)
dense3 = Dense(50)(dense2)
dense4 = Dense(20)(dense3)
dense5 = Dense(10)(dense4)
output1 = Dense(1)(dense5)

model = Model(inputs=input1, outputs=output1)

model.summary()


# model = Sequential()
# model.add(Dense(300, input_dim=13))
# model.add(Dense(100))
# model.add(Dense(50))
# model.add(Dense(20))
# model.add(Dense(10))
# model.add(Dense(1))

# 3. compile
model.compile(loss='mse', optimizer="adam")

model.fit(x_train, y_train, epochs=100, batch_size=10)
# model.fit(x_train, y_train, epochs=100) -> batch_size는 default로 들어가 있다
# 값은 32가 기본값


# 4. evaluate and predict
loss = model.evaluate(x_test, y_test)
print('loss:', loss)

# loss: 37.513221740722656
# loss: 31.852256774902344

y_pred = model.predict(x_test)
print('x_test를 통한 y의 예측값:', y_pred)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred) # y_test, y_pred 차이
print('r2스코어:',r2)

# r2스코어: 0.6144593034548876

# # 시각화 
# plt.scatter(y_test, y_pred)
# plt.plot(x, y_pred, color='red')
# # x와 x_test를 통해 예측한 y의 값을 그래프로 나타낸다
# plt.show()








