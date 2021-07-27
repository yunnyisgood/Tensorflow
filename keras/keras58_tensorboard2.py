from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from tensorflow.keras.callbacks import TensorBoard

# 1. 데이터
x = np.array([1, 2, 4, 3, 5, 6, 7, 8, 9, 10])
y = np.array([1, 2, 4, 3, 5, 6, 7, 8, 9, 10])
print(x)
print(y)
x_pred = [6]

# 6번째 값을 예측할 것 

# 2. model
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(1))


# 3. Compile
model.compile(loss='mse', optimizer="adam")

tb = TensorBoard(log_dir='./_save/_graph', histogram_freq=0,
                    write_graph=True, write_images=True)

model.fit(x, y, epochs=50, batch_size=1, callbacks=tb, validation_split=0.2)

loss = model.evaluate(x, y)
print('loss: ', loss)

# loss:  0.3800013065338135
# loss:  0.38000065088272095
# loss:  0.38007310032844543

result = model.predict(x_pred)
print('6의 예측값: ', result)

# 6의 예측값:  [[5.7026186]]
# 6의 예측값:  [[5.697879]]
# 6의 예측값:  [[5.713823]]
# 6의 예측값:  [[5.682958  5.6843615 5.6852345 5.686938  5.682583 ]]