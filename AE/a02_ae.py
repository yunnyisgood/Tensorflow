'''
AutoEncoder
: 높은 차원(high-dimensional space)을 낮은 차원(low-dimensional space)으로 변경시키는 과정에서 
원본 데이터의 의미 있는 속성들(meaningful properties)을 추출하는 것

'''

from enum import auto
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input
import matplotlib.pyplot as plt

import random

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 784).astype('float')/255
x_test = x_test.reshape(10000, 784).astype('float')/255

def autocoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, input_shape=(784, ),
                        activation='relu')) # dense에서 output은 units
    model.add(Dense(units=784, activation='sigmoid'))
    return model

# model = autocoder(hidden_layer_size=154) 
# model = autocoder(hidden_layer_size=64) 
model = autocoder(hidden_layer_size=16) 


model.compile(optimizer='adam', loss='mse')

model.fit(x_train, x_train, epochs=10)

output = model.predict(x_test)

#이미지 5개를 무작위로 고른다 
random_img = random.sample(range(output.shape[0]), 5)

fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) = \
    plt.subplots(2, 5, figsize=(8, 2))

# 원본 이미지를 맨 위에 그린다
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_img[i]].reshape(28, 28), cmap='gray')
    if i == 0:
        ax.set_ylabel("Input", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토인코더가 출력한 이미지를 아래에 그린다 

for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(x_test[random_img[i]].reshape(28, 28), cmap='gray')
    if i == 0:
        ax.set_ylabel("Output", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()


