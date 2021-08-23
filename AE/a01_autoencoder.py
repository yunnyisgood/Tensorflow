'''
AutoEncoder
: 높은 차원(high-dimensional space)을 낮은 차원(low-dimensional space)으로 변경시키는 과정에서 
원본 데이터의 의미 있는 속성들(meaningful properties)을 추출하는 것

'''

from enum import auto
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 784).astype('float')/255
x_test = x_test.reshape(10000, 784).astype('float')/255

input_img = Input(shape=(784, ))
encoded = Dense(1024, activation='relu')(input_img)
# encoded = Dense(64, activation='relu')(input_img)

decoded = Dense(784, activation='sigmoid')(encoded)
# decoded = Dense(784, activation='relu')(encoded)-> sigmoid 보다 값의 범위가 다양해지기 때문에 훨씬 성능이 안좋다 
# decoded = Dense(784, activation='linear')(encoded)
# decoded = Dense(784, activation='tanh')(encoded)

autoencoder = Model(input_img, decoded)

# autoencoder.summary()

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
# autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.fit(x_train,x_train, epochs=30, batch_size=128, validation_split=0.2)
# 

decode_img = autoencoder.predict(x_test)

results = autoencoder.evaluate(x_test, decode_img)
# print('results: ', results)

import matplotlib.pyplot as plt

n= 10
plt.figure(figsize=(20, 4))

for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decode_img[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
