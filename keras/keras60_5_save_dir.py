from numpy.core.fromnumeric import size
from tensorflow.keras.datasets import fashion_mnist
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

'''
ImageDataGenerator를 사용한 
데이터 증폭 
'''

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

train_datagen = ImageDataGenerator(
    rescale=1./255,  
    horizontal_flip=True, 
    vertical_flip=False,
    width_shift_range=0.1, # 
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.1,
    shear_range=0.5,
    fill_mode='nearest'  
    )

# xy_train = train_datagen.flow_from_directory(
#     '../_data/cat_and_dog/training_set/training_set',     
#     target_size=(32, 32),     
#     batch_size=8100,       
#     class_mode='binary'  )     

augmented_size = 10
# batch_size의 역할 
# 4만개의 이미지로 나누어지게 된다

print(x_train.shape[0]) # 60000

randintx = np.random.randint(x_train.shape[0], size=augmented_size)
# 0부터 x_train.shape[0] 즉 6만까지의 범위에서 size만큼의 개수의 정수를 랜덤하게 생성 

print(x_train[0].shape) # (28, 28)
print(x_train.shape[0]) # 60000
print(randintx) # [48072 39797 41718 ... 31965 27089 48411]
print(randintx.shape) # (40000,)
print(x_train.shape)

x_augmented = x_train[randintx].copy()
y_augmented = y_train[randintx].copy()

print(x_augmented.shape) # (40000, 28, 28)
print(x_augmented[0].shape) # (28, 28) -> 40000개의 데이터가 28, 28 형태로 배치되어 있다 
print(x_augmented.shape[0]) # 40000 -> 데이터의 개수는 40000

'''
ValueError: ('Input data in `NumpyArrayIterator` should have rank 4. 오류 해결하려면?
-> 4차원으로 shape을 바꿔준다 
'''

x_augmented = x_augmented.reshape(x_augmented.shape[0], 28, 28, 1)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)


print(x_augmented.shape, x_train.shape, x_test.shape)
# (10, 28, 28, 1) (60000, 28, 28, 1) (10000, 28, 28, 1)

x_augmented = train_datagen.flow(x_augmented, np.zeros(augmented_size),
                                    batch_size=augmented_size, shuffle=False,
                                    save_to_dir='d:/temp/') # .next()[0]
'''  save_to_dir : 이미지 저장. 교육용 데이터가 아닌 일반 사진일 경우 최소 100, 100 크기로 저장해야 한다  '''

# 1개의 데이터가 4만개로 증폭되는 것이 아니라, 4만개의 데이터가                                     
print(x_augmented[0][0].shape) # (40000, 28, 28, 1)
print(x_augmented[0][1].shape) # (40000,)
print(x_augmented[0][0][:10])


# next() 후에 
'''print(x_augmented[0][0].shape) # (28, 1)
print(x_augmented[0][1].shape) # (28, 1)
print(x_augmented.shape) # (40000, 28, 28, 1)
print(x_augmented[0].shape) # (28, 28, 1)'''

x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))

print(x_train.shape, y_train.shape)
# (100000, 28, 28, 1) (100000,)

print(x_train.shape[:10])

'''
x_augument 10개와 원래 x_train 10개를 비교하는 이미지를 출력할 것 
subplot (2, 10, ?) 사용
2시까지

x_train[]
x_augement[]

'''

import matplotlib.pyplot as plt

# plt.figure(figsize=(7, 7))
# for i in range(49):
#     plt.subplot(7, 7, i+1)
#     plt.axis('off')
#     plt.imshow(x_train[0][i], cmap='gray')
# plt.show()

# for i in range(11):
#     plt.figure(figsize=(2,10))
#     plt.subplot (2, 10, i+1)
#     plt.imshow(x_train[0][i], cmap='gray')
# plt.show()