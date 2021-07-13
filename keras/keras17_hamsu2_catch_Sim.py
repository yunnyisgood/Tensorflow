from numpy.core.fromnumeric import shape
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
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
input1 = Input(shape=(1,))
dense1 = Dense(5)(input1)
dense2 = Dense(4)(dense1)
dense3 = Dense(6)(dense2)
dense4 = Dense(5)(dense3)
dense5 = Dense(4)(dense4)
dense6 = Dense(4)(dense5)
dense7 = Dense(5)(dense6)
dense8 = Dense(3)(dense7)
dense9 = Dense(5)(dense8)
dense10 = Dense(1)(dense9)
dense11 = Dense(3)(dense10)
dense12 = Dense(4)(dense11)
dense13 = Dense(5)(dense12)
dense14 = Dense(3)(dense13)
dense15 = Dense(5)(dense14)
dense16 = Dense(1)(dense15)
dense17 = Dense(5)(dense16)
dense18 = Dense(3)(dense17)
dense19 = Dense(4)(dense18)
dense20 = Dense(3)(dense19)
dense21 = Dense(6)(dense20)
dense22 = Dense(5)(dense21)
output1 = Dense(1)(dense22)

model = Model(inputs=input1, outputs=output1)

model.summary()

# model = Sequential()
# model.add(Dense(5, input_dim=1))
# model.add(Dense(4))
# model.add(Dense(6))
# model.add(Dense(5))
# model.add(Dense(4))
# model.add(Dense(4))
# model.add(Dense(5))
# model.add(Dense(3))
# model.add(Dense(5))
# model.add(Dense(1))
# model.add(Dense(3))
# model.add(Dense(4))
# model.add(Dense(5))
# model.add(Dense(3))
# model.add(Dense(5))
# model.add(Dense(1))
# model.add(Dense(5))
# model.add(Dense(3))
# model.add(Dense(4))
# model.add(Dense(5))
# model.add(Dense(3))
# model.add(Dense(6))
# model.add(Dense(5))
# model.add(Dense(1))


# 3. Compile
model.compile(loss='mse', optimizer="adam")

# model.fit(x, y, epochs=5000, batch_size=1)
# model.fit(x, y, epochs=3000, batch_size=1)
model.fit(x, y, epochs=3000, batch_size=100)

loss = model.evaluate(x, y)
print('loss: ', loss)

# loss:  0.3800013065338135
# loss:  0.38000065088272095
# loss:  0.38007310032844543
# loss:  0.37999990582466125


y_pred = model.predict(x)
print('x의 예측값: ', y_pred)


from sklearn.metrics import r2_score
r2 = r2_score(y, y_pred) # y_test, y_pred 차이

print('r2스코어:',r2)

# r2스코어: 0.8100000762938692
# r2스코어: 0.8100000429152772


# 과제 2
# R2를 0.9 올려라
# 기한은 일요일 밤 12시까지