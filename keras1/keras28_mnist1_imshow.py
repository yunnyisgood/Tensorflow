import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

print(x_train[111])
print("y[0] 값: ", y_train[111]) # 5 -> 왜 값 1개로 나오지?

plt.imshow(x_train[111], 'gray')
plt.show()