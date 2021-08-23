'''
2번을 카피해서
딥하게 구성한다 
2개의 모델을 구성하는데 하나는 기본적 오토인코더
다른 하나는 딥하게 구성 

=> 두개의 모델 layer차이 크게 주기 
==> 그래프 다시!
'''

from enum import auto
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input
import matplotlib.pyplot as plt

import random

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 784).astype('float')/255
x_test = x_test.reshape(10000, 784).astype('float')/255

def autocoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, input_shape=(784, ),
                        activation='relu')) # dense에서 output은 units
    model.add(Dense(units=784, activation='sigmoid'))
    return model

def autocoder2(hidden_layer_size):
    model = Sequential()
    model.add(Dense(5000, input_shape=(784, ),
                        activation='relu')) 
    model.add(Dense(5000, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(units=784, activation='sigmoid'))
    return model
# model = autocoder(hidden_layer_size=154) 
# model = autocoder(hidden_layer_size=64) 
model = autocoder(hidden_layer_size=32) 
model2 = autocoder2(hidden_layer_size=64)

model.compile(optimizer='adam', loss='mse')
model2.compile(optimizer='adam', loss='mse')

model.fit(x_train, x_train, epochs=10)
model2.fit(x_train, x_train, epochs=10)

output = model.predict(x_test)
output2 = model2.predict(x_test)

#이미지 5개를 무작위로 고른다 
random_img = random.sample(range(output.shape[0]), 5)
random_img2 = random.sample(range(output2.shape[0]), 5)

fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) = \
    plt.subplots(2, 5, figsize=(8, 2))

# 1. 기본적 autoencoder
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

fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) = \
    plt.subplots(2, 5, figsize=(8, 2))

# 2. 딥한 형태
# 원본 이미지를 맨 위에 그린다
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_img2[i]].reshape(28, 28), cmap='gray')
    if i == 0:
        ax.set_ylabel("Input2", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])


# 오토인코더가 출력한 이미지를 아래에 그린다 

for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(x_test[random_img2[i]].reshape(28, 28), cmap='gray')
    if i == 0:
        ax.set_ylabel("Output2", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()


