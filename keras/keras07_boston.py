from sklearn.datasets import load_boston
from tensorflow.python.keras.models import Sequential # 교육용 예제 임포트
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#1. data
datasets = load_boston()
x = datasets.data
y = datasets.target

print(x.shape) # (506, 13) -> 13가지의 특성 -> Input_dim = 13
print(y.shape) # (506,) -> output_dim = 1
print(datasets.DESCR)

print(datasets.feature_names)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)

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

model.fit(x_train, y_train, epochs=1000, batch_size=10)

# 4. evaluate and predict
loss = model.evaluate(x_test, y_test)
print('loss:', loss)

y_pred = model.predict(x_test)
print('y의 예측값:', y_pred)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred) # y_test, y_pred 차이
print('r2스코어:',r2)









