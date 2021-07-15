# EarlyStopping -> 학습하는 과정에서 어느정도 loss기 떨어졌을 경우 빨리 멈춘다
# epochs를 많이 줘도 최저점을 맞춰서 earlyStopping을 주면 빨리 멈춰진다...

from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer, StandardScaler, MinMaxScaler
from sklearn.datasets import load_boston 
from tensorflow.python.keras.models import Sequential 
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


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

es = EarlyStopping(monitor='loss', patience=20, mode='min', verbose=1)
# loss의 갱신값으로 모니터, loss값이 갱신될때까지 몇번 참을것인지가 patience
# loss 의 경우, 최소화 시키는 방향으로 training 이 진행되므로 min 을 지정

hist = model.fit(x_train, y_train, epochs=1000, batch_size=8,
                validation_split=0.2,  callbacks=[es])

print(hist.history.keys()) # dict_keys(['loss', 'val_loss']) 
# -> loss, val_loss를 dict 형태로 반환하겠다는 의미
print('*'*100)
print(hist.history['loss'])
print('*'*100)
print(hist.history['val_loss'])

print('*'*100)


# 4. evaluate and predict
loss = model.evaluate(x_test, y_test) 
# 이미 나와있는 weight 에 평가만 해준다
print('loss:', loss)

y_pred = model.predict(x_test)
# print('x_test를 통한 y의 예측값:', y_pred)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred) # y_test, y_pred 차이
print('r2스코어:',r2)

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])

plt.title("loss, val_loss")
plt.xlabel("epochs")
plt.ylabel("loss, val_loss")
plt.legend(["train loss", "val_loss"])
plt.show()


# verbose 지정 안해줬을 때
# Epoch 34/100
# 45/45 [==============================] - 0s 3ms/step - loss: 26.4376
# 5/5 [==============================] - 0s 0s/step - loss: 21.5655
# loss: 21.565500259399414

#verbose=1 일 때
# Epoch 26/100
# 45/45 [==============================] - 0s 3ms/step - loss: 43.8210
# Epoch 00026: early stopping
# 5/5 [==============================] - 0s 0s/step - loss: 35.0108
# loss: 35.010780334472656


