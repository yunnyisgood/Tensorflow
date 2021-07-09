# 1. R2를 음수가 아닌 0.5 이하로 만들어라.
# 2. 데이터 변경 x
# 3. layer -> Input, output 포함 6개 이상
# 4. batch_size = 1
# 5. epochs는 100 이상
# 6. hidden layer의 node는 10<= <=1000개 이하
# 7. train 70%

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 1. 데이터
x = np.array(range(100)) 
y = np.array(range(1,101))

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)

print(x_train.shape, y_train.shape)  # (70,) (70,)
print(x_test.shape, y_test.shape)  # (30,) (30,)

# 2. model
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(1))

# 3. Compile
model.compile(loss='kld', optimizer='adam')

model.fit(x_train, y_train, epochs=300, batch_size=1)

# 4. evaluate and predict
loss = model.evaluate(x_test, y_test)
print('loss:', loss)

y_pred = model.predict(x_test)
print('y의 예측값:', y_pred)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print('r2의 스코어: ', r2)

# r2의 스코어:  0.9999999999831506

