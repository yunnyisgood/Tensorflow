import numpy as np
from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer, StandardScaler, MinMaxScaler
from tensorflow.python.keras.models import Sequential 
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split


x_data = np.load('./_save/_npy/k55_x_data_boston.npy')
y_data = np.load('./_save/_npy/k55_y_data_boston.npy')

print(x_data)
print(y_data)
print(x_data.shape, y_data.shape)
# (506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, 
train_size=0.7, shuffle=True, random_state=9)

scaler = PowerTransformer()
scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)


# 2.model
model = Sequential()
model.add(Dense(1000, input_dim=13, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))


# 3. compile
model.compile(loss='mse', optimizer="adam")

model.fit(x_train, y_train, epochs=100, batch_size=8)
# model.fit(x_train, y_train, epochs=100) -> batch_size는 default로 들어가 있다
# 값은 32가 기본값


# 4. evaluate and predict
loss = model.evaluate(x_test, y_test)
print('loss:', loss)

# loss: 37.513221740722656
# loss: 17.039806365966797

y_pred = model.predict(x_test)
# print('x_test를 통한 y의 예측값:', y_pred)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred) # y_test, y_pred 차이
print('r2스코어:',r2)

'''
PowerTransformer, MinMaxScaler, StandardScaler 순서로
가장 성능이 우수
 
'''

'''
loss: 24.98827362060547
r2스코어: 0.7184801979372041
'''