from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np


# 완성한뒤, 출력결과 스샷

# 1. 데이터
x = np.array([1, 2, 4, 3, 5])
y = np.array([1, 2, 3, 4, 5])
print(x)
print(y)
x_pred = [6]

# 6번째 값을 예측할 것 

# 2. model
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(4))
model.add(Dense(6))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(5))
model.add(Dense(1))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(5))
model.add(Dense(1))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(6))
model.add(Dense(5))
model.add(Dense(1))


# 3. Compile
model.compile(loss='mse', optimizer="adam")

# model.fit(x, y, epochs=5000, batch_size=1)
# model.fit(x, y, epochs=3000, batch_size=1)
model.fit(x, y, epochs=3000, batch_size=150)

loss = model.evaluate(x, y)
print('loss: ', loss)

# loss:  0.3800013065338135
# loss:  0.38000065088272095
# loss:  0.38007310032844543


y_pred = model.predict(x)
print('x의 예측값: ', y_pred)


from sklearn.metrics import r2_score
r2 = r2_score(y, y_pred) # y_test, y_pred 차이

print('r2스코어:',r2)

# r2스코어: 0.8100000762938692


# 과제 2
# R2를 0.9 올려라
# 기한은 일요일 밤 12시까지