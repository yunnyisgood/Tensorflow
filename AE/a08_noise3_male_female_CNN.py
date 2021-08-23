'''
실습, 과제
기존 데이터에 noise를 넣어서
기미 주근깨 여드름을 제거하시오 
'''

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import time
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, QuantileTransformer, StandardScaler
import matplotlib.pyplot as plt
import random
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten

x_train = np.load('./_save/kerask59_men_women_x_train.npy')
y_train = np.load('./_save/kerask59_men_women_y_train.npy')
x_test = np.load('./_save/kerask59_men_women_x_test.npy')
y_test = np.load('./_save/kerask59_men_women_y_test.npy')
x_pred = np.load('D:/Tensorflow/_save/kerask59_men_women_mine2.npy')

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape, x_pred.shape)
# (2486, 80, 80, 3) (2483,) (828, 80, 80, 3) (827,) (5, 80, 80, 3)

x_train = x_train.reshape(2486, 80 * 80*3)
x_test = x_test.reshape(828, 80 * 80*3)
x_pred = x_pred.reshape(5, 80*80*3)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(2486, 80 , 80, 3).astype('float')/255
x_test = x_test.reshape(828, 80 , 80, 3).astype('float')/255
x_pred = x_pred.reshape(5, 80 , 80, 3).astype('float')/255


x_train_noised = x_train + np.random.normal(0, 0.1, size=x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.1, size=x_test.shape)
x_pred_noised = x_pred + np.random.normal(0, 0.1, size=x_pred.shape)

#값을 0~1 사이의 값으로 다시 변환
x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)
x_pred_noised = np.clip(x_pred_noised, a_min=0, a_max=1)

# modeling
def autocoder():
    input = Input(shape=(80, 80, 3))
    encoded = Conv2D(16,(2, 2), activation='relu')(input)
    encoded = MaxPooling2D((2, 2))(encoded)
    encoded = Conv2D(32,(2, 2), activation='relu')(encoded)
    encoded = MaxPooling2D((2, 2))(encoded)
    encoded = Conv2D(64,(2, 2), activation='relu')(encoded)

    decoded = Conv2D(32,(2, 2), activation='relu')(encoded)
    decoded = UpSampling2D(size=(2, 2))(decoded)
    decoded = Conv2D(16,(2, 2), activation='relu')(decoded)
    decoded = UpSampling2D(size=(2, 2))(decoded)
    output = Conv2D(3,(2,2))(decoded)
    autoencoder = Model(input, output)
    return autoencoder


model = autocoder() # pca의 95%지점의 개수 => 실제와 유사하게 출력
model.summary()

model.compile(optimizer='adam', loss='mse')

model.fit(x_train_noised, x_train, epochs=3, batch_size=8)    

output = model.predict(x_pred_noised)
output2 = model.predict([x_pred_noised][-1])

# print('여자일 확률 : ', output2, "%")

random_img = random.sample(range(output.shape[0]), 3)

fig, ((ax1, ax2, ax3), (ax6, ax7, ax8), (ax11, ax12, ax13)) = \
    plt.subplots(3, 3, figsize=(20, 8))

# 첫번째 행에는 원본 이미지
for i, ax in enumerate([ax1, ax2, ax3]):
    ax.imshow(x_pred[random_img[i]].reshape(80, 80, 3), cmap = 'gray')
    if i == 0:
        ax.set_ylabel("Input", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

#  두번째 행에는 noise
for i, ax in enumerate([ax11, ax12, ax13]):
    ax.imshow(x_pred_noised[random_img[i]].reshape(80, 80, 3))
    if i == 0:
        ax.set_ylabel("noise", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

#  세번째 행에는 autoencoder
for i, ax in enumerate([ax6, ax7, ax8]):
    ax.imshow(x_pred[random_img[i]].reshape(80, 80, 3))
    if i == 0:
        ax.set_ylabel("Output", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()