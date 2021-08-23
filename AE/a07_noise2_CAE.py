'''
-> Conv2d로 실습 

'''

from enum import auto
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input
import matplotlib.pyplot as plt
import random
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 28, 28, 1).astype('float')/255
x_test = x_test.reshape(10000,  28, 28, 1).astype('float')/255

# noise 생성
x_train_noised = x_train + np.random.normal(0, 0.1, size=x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.1, size=x_test.shape)
# 0.1보다 큰 값을 넣을수록 노이즈가 더 커진다 
# 이렇게 만들경우 이미지 최댓값이 리밋인 255를 초과하게 됨

#값을 0~1 사이의 값으로 다시 변환
x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)

# modeling
def autocoder2():
    input = Input(shape=(28, 28, 1))
    encoded = Conv2D(1024,(2, 2), activation='relu', padding='same')(input)
    encoded = MaxPooling2D((1, 1), padding='same')(encoded)
    encoded = Conv2D(128,(2, 2), activation='relu', padding='same')(encoded)
    encoded = MaxPooling2D((1, 1), padding='same')(encoded)
    encoded = Conv2D(64,(2, 2), activation='relu', padding='same')(encoded)

    decoded = Conv2D(1024,(2, 2), activation='relu', padding='same')(encoded)
    decoded = UpSampling2D(size=(1, 1))(decoded)
    decoded = Conv2D(1024,(2, 2), activation='relu', padding='same')(decoded)
    decoded = UpSampling2D(size=(1, 1))(decoded)
    output = Conv2D(1,(2,2), activation='sigmoid', padding='same')(decoded)
    autoencoder = Model(input, output)
    return autoencoder

model = autocoder2() # pca의 95%지점의 개수 => 실제와 유사하게 출력

model.compile(optimizer='adam', loss='mse')


model.fit(x_train_noised, x_train, epochs=3, batch_size=128)

output = model.predict(x_test_noised)

#이미지 5개를 무작위로 고른다 
random_img = random.sample(range(output.shape[0]), 5)

fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10), (ax11, ax12, ax13, ax14, ax15)) = \
    plt.subplots(3, 5, figsize=(20, 8))

# 첫번째 행에는 원본 이미지
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_img[i]].reshape(28, 28), cmap='gray')
    if i == 0:
        ax.set_ylabel("Input", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

#  두번째 행에는 noise
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(x_test_noised[random_img[i]].reshape(28, 28), cmap='gray')
    if i == 0:
        ax.set_ylabel("noise", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

#  세번째 행에는 autoencoder
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(x_test[random_img[i]].reshape(28, 28), cmap='gray')
    if i == 0:
        ax.set_ylabel("Conv2D", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()


