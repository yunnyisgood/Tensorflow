from tensorflow.keras.datasets import fashion_mnist
import numpy as np

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=False,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.1,
    shear_range=0.5,
    fill_mode='nearest',
    )

#1. ImageDataGenerator를 정의
#2. 파일에서 땡겨오려면 -> flow_from_directory()  :  xy 가 튜플형태로 묶여서 나옴
#3. 데이터에서 땡겨오려면 -> flow()  :  x와 y가 분류되어 있어야 한다.

augment_size=10

randidx = np.random.randint(x_train.shape[0], size=augment_size)       # x_train[0]에서 아그먼트 사이즈 만큼 랜덤하게 들어감

print(x_train.shape[0])     # 60000
print(randidx)              # [44596 49164  1092 ... 51768  3501 13118]
print(randidx.shape)        # (40000,)

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()

print(x_augmented.shape)       # (40000, 28, 28)
# flow 의 x는 4차원을 받아야 한다!!

x_augmented = x_augmented.reshape(x_augmented.shape[0], 28, 28, 1)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

import time
start = time.time()
                                 # x          # y (y_argumented 넣어도 됨)
x_augmented = train_datagen.flow(x_augmented, np.zeros(augment_size), 
batch_size=augment_size, shuffle=False,   # flow에서 출력 : iterator
                                save_to_dir='d:/temp/')#.next()[0]   # save_to_dir: 이경로에 저장 됨(이미지 보고 싶을  때)
end = time.time() - start
print(x_augmented[0][0].shape)       # (40000, 28, 28, 1)
print(x_augmented[0][1].shape)  
print(x_augmented[0][1][:10])  
print(x_augmented[0][1][10:15])  
print('걸린시간 :', end)
# x_train = np.concatenate((x_train, x_argumented))
# y_train = np.concatenate((y_train, y_argumented))
# print(x_train.shape, y_train.shape)     # (100000, 28, 28, 1) (100000,)
# x_argumented = x_augmented[:10,:,:,:]
# x_train = x_train[:10,:,:,:]                # x_argumented.shape: (10, 28, 28, 1)
# ic(x_argumented.shape, x_train.shape)       # x_train.shape: (10, 28, 28, 1)
# import matplotlib.pyplot as plt
# plt.figure(figsize=(10, 2))
# for i in range(20):
#     plt.subplot(2, 10, i+1)
#     plt.axis('off')
#     if i < 10:
#         plt.imshow(x_train[i], cmap='gray')
#     else:
#         plt.imshow(x_argumented[i-10], cmap='gray')
# plt.show()