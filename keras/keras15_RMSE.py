from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 1. 데이터
x = np.array(range(100))
y = np.array(range(1,101))

# 인덱싱 방법 1
# x_train = x[0:70]
# y_train = y[:70]
# x_test = x[-30:]  # x[7:0]과 동일
# y_test = y[70:]


# 인덱싱 방법 2
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)
#  shuffle은 기본값이 True이므로 굳이 적어주지 않아도 된다.
# random_state 는 재현가능(for reproducibility)하도록 난수의 초기값을 설정해주는 것 -> 아무 숫자나 넣어주면 된다.

print(x_train.shape, y_train.shape)  # (70,) (70,)
print(x_test.shape, y_test.shape)  # (30,) (30,)


# 2. model
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(4))
model.add(Dense(5))
model.add(Dense(7))
model.add(Dense(3))
model.add(Dense(1))


# 3. Compile
model.compile(loss='mse', optimizer="adam")
print(x_train.shape, y_train.shape)  # (70,) (70,)
print(x_test.shape, y_test.shape)  # (30,) (30,)

model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)

y_pred = model.predict(x_test)  # 훈련시킨 weight를 넣어 예측값을 도출해낸다. 
# 비교값은 y_test, y_predict 
# 원래 y값, 훈련시켜서 도출해낸 결과값
print('x_test의 예측값:', y_pred)

# loss:  6.36716768198653e-09

from sklearn.metrics import r2_score, mean_squared_error
r2 = r2_score(y_test, y_pred) # y_test, y_pred 차이
print('r2스코어:',r2)

# r2스코어: 0.999999999992718

def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))
rmse = RMSE(y_test, y_pred)
print('rmse: ', rmse)

# rmse:  7.979453471763823e-05
# mse의 경우 에러의 제곱이므로, 에러가 커질수록 그에 따란 가중치가 높이 반영된다.
# 가중치가 높이 반영되는 것을 방지하기 위해 rmse는 mse 값의 루트를 씌운다