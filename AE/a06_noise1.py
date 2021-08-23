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

# noise 생성
x_train_noised = x_train + np.random.normal(0, 0.1, size=x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.1, size=x_test.shape)
# 0.1보다 큰 값을 넣을수록 노이즈가 더 커진다 
# 이렇게 만들경우 이미지 최댓값이 리밋인 255를 초과하게 됨

#값을 0~1 사이의 값으로 다시 변환
x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)

# modeling
def autocoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, input_shape=(784, ),
                        activation='relu')) # dense에서 output은 units
    model.add(Dense(units=784, activation='sigmoid'))
    return model

model = autocoder(hidden_layer_size=154) # pca의 95%지점의 개수 => 실제와 유사하게 출력
# model = autocoder(hidden_layer_size=64) 
# model = autocoder(hidden_layer_size=16) 

model.compile(optimizer='adam', loss='mse')

model.fit(x_train_noised, x_train, epochs=10)
# 훈련 시킬 때는 noise 존재한 값, noise 존재하지 않는 값 두개 모두 훈련 시켜야 함
# 그리고 test 할 때에는 noise 존재하는 값을 넣었을 때 noise 없는 값으로 예측하여 출력된다 

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
        ax.set_ylabel("Output", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()


