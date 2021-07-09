import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt


# 1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],[1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3]]) # (2, 10)
x = np.transpose(x) # (10, 2) 
y = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20]) #(10, )
y = np.transpose(y)
x_pred = np.array([[10, 1.3]]) # (1, 2)  => 행무시, 열우선으로 열만 맞춰주면 된다 

# 2. model
model = Sequential()
model.add(Dense(1, input_dim=2)) # output_dim = 
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(5))
model.add (Dense(1))

# 3. compile
model.compile(loss="mse", optimizer="adam")

model.fit(x, y, epochs=1000, batch_size=1)

#4. 평가 예측
loss = model.evaluate(x1, y)
print('loss: ', loss)

# loss:  0.11791887134313583

result = model.predict(x_pred)
print('result: ', result)

# result:  [[19.25796]]

y_pred = model.predict(x)
plt.scatter(x[:,0], y)
plt.scatter(x[:,1], y)
plt.plot(x, y_pred, color='red')
plt.show


