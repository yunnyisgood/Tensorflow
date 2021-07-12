import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import time


# 1. 데이터
x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
             [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],
             [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]])
print(x.shape)
x = np.transpose(x) # (10, 3)
print(x.shape)
y = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20]) # (10,)
print(y.shape)
x_pred = np.array([[10, 1.3, 1]]) 
print(x_pred.shape)


# 2. model
model = Sequential()
model.add(Dense(5, input_dim=3))
model.add(Dense(4))
model.add(Dense(5))
model.add(Dense(7))
model.add(Dense(2))
model.add(Dense(1))

# 3. compile
model.compile(loss="mse", optimizer="adam")

start = time.time()
model.fit(x, y, epochs=100, batch_size=1, verbose=0)
end = time.time() - start
print("걸린시간 : ", end)


# verbose
# 0
# 걸린시간: 2.218069076538086

# 1
# 걸린시간: 2.934088945388794

# 2
# 걸린시간: 2.4624149799346924

# 3 
# 걸린시간:2.470020055770874


#4. 평가 예측
loss = model.evaluate(x, y)
print('loss: ', loss)

# loss:  3.352187923155725e-05

result = model.predict(x_pred)
print('result: ', result)

# result:  [[  3.7012   28.30966 208.91016]]

y_pred = model.predict(x)
plt.scatter(x[:,0], y)
plt.scatter(x[:,1], y)
plt.scatter(x[:,2], y)
plt.plot(x, y_pred, color='red')
plt.show()