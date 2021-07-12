from sklearn.datasets import load_boston # 교육용 예제 임포트
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
# print(datasets.DESCR)

print(datasets.feature_names)

x_train, x_test, y_train, y_test = train_test_split(x, y, 
train_size=0.7, shuffle=True, random_state=66)

# 2.model
model = Sequential()
model.add(Dense(10, input_dim=13))
model.add(Dense(7))
model.add(Dense(5))
model.add(Dense(9))
model.add(Dense(4))
model.add(Dense(8))
model.add(Dense(3))
model.add(Dense(5))
model.add(Dense(1))


# 3. compile
model.compile(loss='mse', optimizer="adam")

model.fit(x_train, y_train, epochs=100, batch_size=43)


# 4. evaluate and predict
loss = model.evaluate(x_test, y_test)
print('loss:', loss)

# loss: 76.62830352783203

y_pred = model.predict(x_test)
print('x_test를 통한 y의 예측값:', y_pred)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred) # y_test, y_pred 차이
print('r2스코어:',r2)

# r2스코어: 0.07248870680689357

# 시각화 
plt.scatter(y_test, y_pred)
plt.plot(x, y_pred, color='red')
# x와 x_test를 통해 예측한 y의 값을 그래프로 나타낸다
plt.show()








