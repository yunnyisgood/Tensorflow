import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

'''
batch_size를 크게 주어서 하나로 통으로 엮여있도록 해준다 
xy_train, xy_test[0]에 몰아서 주기 

'''

train_datagen = ImageDataGenerator(
    rescale=1./255,  # 0~255로 되어 있는 데이터를 255로 나누어준다 
    horizontal_flip=True, # 수평이동 유무
    vertical_flip=True,
    width_shift_range=0.1, # 
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2, # 원래 이미지에서 20% 정도 더 확대해서
    shear_range=0.7,
    fill_mode='nearest'  #공백을 근접하게 채우겠다는 의미
    )

test_datagen = ImageDataGenerator(rescale=1./255)

xy_train = train_datagen.flow_from_directory(# x와 y가 동시에 생성됨
    '../_data/brain/train',      # 이미지가 있는 폴더 지정이 아닌 상위 폴더(동일한 급의 라벨이 모여있는 폴더까지)로 지정   # train(ad/normal)
    target_size=(150, 150),     # 임의대로 크기 조정
    batch_size=200,        # y 하나의 개수
    class_mode='binary'         # 이상이 있다-라벨 / 이상이 없다-라벨 : 이진분류
)
# Found 160 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    '../_data/brain/test',      # 이미지가 있는 폴더 지정이 아닌 상위 폴더(동일한 급의 라벨이 모여있는 폴더까지)로 지정   # train(ad/normal)
    target_size=(150, 150),     # 임의대로 크기 조정
    batch_size=200,        # y 하나의 개수
    class_mode='binary'         # 이상이 있다-라벨 / 이상이 없다-라벨 : 이진분류
)
# Found 120 images belonging to 2 classes.

print(xy_train[0][0].shape, xy_train[0][1].shape)   # (160, 150, 150, 3) (160,)
print(xy_test[0][0].shape, xy_test[0][1].shape) # (120, 150, 150, 3) (120,)

np.save('./_save/_npy/kerask59_3_train_x.npy', arr=xy_train[0][0])
np.save('./_save/_npy/kerask59_3_train_y.npy', arr=xy_train[0][1])
np.save('./_save/_npy/kerask59_3_test_x.npy', arr=xy_test[0][0])
np.save('./_save/_npy/kerask59_3_test_y.npy', arr=xy_test[0][1])