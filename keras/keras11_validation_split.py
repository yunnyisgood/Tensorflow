from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 1. 데이터
# 훈련 데이터 
x = np.array([1, 2, 4, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13])
y = np.array([1, 2, 4, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13])

x_train, x_test, y_train, y_test = train_test_split(x, y, 
train_size=0.8, shuffle=True, random_state=66)
# 15개의 데이터 중에서 80% -> train data는 12개



# 2. model
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(4))
model.add(Dense(5))
model.add(Dense(7))
model.add(Dense(3))
model.add(Dense(1))


# 3. Compile
model.compile(loss='mse', optimizer="adam", metrics=['mae'])

# model.fit(x_train, y_train, epochs=1000, batch_size=1,verbose=1, validation_data=(x_val, y_val))
# 훈련할 때 로직 하나 추가 
model.fit(x_train, y_train, epochs=1000, batch_size=1, validation_split=0.2)
# 12개의 train 데이터 중에서 validation data는 3.6개
# 실제 train 데이터는 8.4개 -> 반올림해서 8개


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)

# loss:  3.2621681690216064

# result = model.predict([x_pred])
# print('10의 예측값: ', result)

# 10의 예측값:  [[9.276111]]

y_pred = model.predict(x_test)
print('y예측값:', y_pred)
# plt.scatter(x, y)
# plt.plot(x, y_pred, color='red')
# plt.show()