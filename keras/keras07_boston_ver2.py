from keras.datasets import boston_housing
from tensorflow.python.keras.models import Sequential 
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np

# data

(x_train, y_train), (x_test, y_test)= boston_housing.load_data()

# 정규화 과정 추가

mean = x_train.mean(axis = 0)
std = x_train.std(axis = 0)
x_train = (x_train - mean)/std  # 훈련데이터 
x_test = (x_test - mean)/std # 검증데이터 

print(x_train.shape)  # (404, 13)
print(y_train.shape) # (404, )
print(x_test.shape) # (102, 13)
print(y_test.shape) # (102, )


# modeling
model = Sequential()
model.add(Dense(33, activation='relu', input_shape=(13,)))
model.add(Dense(33, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(1))

# compile
model.compile(optimizer='adam', loss='mse', metrics='mae') # 오류나면 metrics 추가
# 모델 훈련 과정을 측정하는 지표로는 mae를 사용
# mse는 오차를 측정, 프로세스 자체를 측정할 때에는 mae를 사용

# model 구조 요약
model.summary()

model.fit(x_train,y_train, epochs=100, batch_size=30 )


# 4. evaluate and predict
loss = model.evaluate(x_test, y_test) 
print('loss:', loss)

# loss: [19.372793197631836, 2.867462158203125]

y_pred = model.predict(x_test)
print('y의 예측값:', y_pred)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred) # y_test, y_pred 차이
print('r2스코어:',r2)

# r2스코어: 0.7597422462963858

# 예측값과 실제값의 차이 시각화 
plt.scatter(y_test, y_pred)
plt.xlabel('True Values[1000$]')
plt.ylabel('Predictions [1000$]')
plt.axis('equal')
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
_ = plt.plot([-10, 100], [-10, 100])
plt.show()

# 예측 오차 분포 
plt.hist(loss, bins=50)
plt.xlabel('Predictions error [1000$]')
plt.ylabel('Count')
plt.show()
