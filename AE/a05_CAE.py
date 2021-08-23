'''
2번을 카피해서
딥하게 구성한다 
2개의 모델을 구성하는데 하나는 기본적 오토인코더
다른 하나는 cnnn으로 구성 
conv2d, maxpool, conv2d, maxpool, conv2d -> encoder
conv2d, upsampling2d, conv2d, upsampling2d, conv2d(1, ) -> Decoder
'''

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten
import matplotlib.pyplot as plt

import random

(x_train, _), (x_test, _) = mnist.load_data()


def autocoder(hidden_layer_size):

    model = Sequential()
    model.add(Dense(units=hidden_layer_size, input_shape=(784, ),
                        activation='relu')) # dense에서 output은 units
    model.add(Dense(units=784, activation='sigmoid'))
    return model

def autocoder2():
    input = Input(shape=(28, 28, 1))
    encoded = Conv2D(1024,(2, 2), activation='relu', padding='same')(input)
    encoded = MaxPooling2D((2,2), padding='same')(encoded)
    encoded = Conv2D(128,(2, 2), activation='relu', padding='same')(encoded)
    encoded = MaxPooling2D((2,2), padding='same')(encoded)
    encoded = Conv2D(64,(2, 2), activation='relu', padding='same')(encoded)

    decoded = Conv2D(1024,(2, 2), activation='relu', padding='same')(encoded)
    decoded = UpSampling2D(size=(2,2))(decoded)
    decoded = Conv2D(1024,(2, 2), activation='relu', padding='same')(decoded)
    decoded = UpSampling2D(size=(2,2))(decoded)
    output = Conv2D(1,(2,2), activation='sigmoid', padding='same')(decoded)
    autoencoder = Model(input, output)
    return autoencoder


model2 = autocoder2()
model2.compile(optimizer='adam', loss='binary_crossentropy')

x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
model2.fit(x_train, x_train, epochs=10, batch_size=128)
output2 = model2.predict(x_test)


x_train = x_train.reshape(60000, 784).astype('float')/255
x_test = x_test.reshape(10000, 784).astype('float')/255
model = autocoder(hidden_layer_size=32) 

model.compile(optimizer='adam', loss='mse')

model.fit(x_train, x_train, epochs=10, batch_size=128)

output = model.predict(x_test)

# 그래프 그리기 -> 다시

fig, axes = plt.subplots(7, 5, figsize=(15, 15))

random_imgs = random.sample(range(output.shape[0]), 5)
random_imgs2 = random.sample(range(output2.shape[0]), 5)

fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) = \
    plt.subplots(3, 5, figsize=(20, 8))

# 첫번째 행에는 원본 이미지
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_imgs[i]].reshape(28, 28), cmap='gray')
    if i == 0:
        ax.set_ylabel("Input", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(x_test[random_imgs2[i]].reshape(28, 28), cmap='gray')
    if i == 0:
        ax.set_ylabel("Output", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()
