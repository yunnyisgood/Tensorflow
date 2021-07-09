from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 1. 데이터
x = np.array(range(100))
y = np.array(range(1,101))

x_train = x[0:70]
y_train = y[:70]
x_test = x[-30:]  # x[7:0]과 동일
y_test = y[70:]

print(x_train.shape, y_train.shape)  # (70,) (70,)
print(x_test.shape, y_test.shape)  # (30,) (30,)

# np.random.shuffle(x_train)
# np.random.shuffle(y_train)
# np.random.shuffle(x_test)
# np.random.shuffle(y_test)

x_train, y_train, x_test, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)
#  shuffle은 기본값이 True이므로 굳이 적어주지 않아도 된다.
# random_state 는 재현가능(for reproducibility)하도록 난수의 초기값을 설정해주는 것 -> 아무 숫자나 넣어주면 된다.

print(x_train)
print(y_train)
print(x_test)
print(y_test)


# # 2. model
# model = Sequential()
# model.add(Dense(5, input_dim=1))
# model.add(Dense(4))
# model.add(Dense(5))
# model.add(Dense(7))
# model.add(Dense(3))
# model.add(Dense(1))


# # 3. Compile
# model.compile(loss='mse', optimizer="adam")

# model.fit(x_train, y_train, epochs=1000, batch_size=1)

# #4. 평가, 예측
# loss = model.evaluate(x_test, y_test)
# print('loss: ', loss)

# # loss:  3.2621681690216064

# # result = model.predict([x_pred])
# # print('10의 예측값: ', result)

# # 10의 예측값:  [[9.276111]]

# y_pred = model.predict([11])
# # plt.scatter(x, y)
# # plt.plot(x, y_pred, color='red')
# # plt.show()