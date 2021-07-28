import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
### ImageDataGenerator - test 데이터는 Generator로 증폭시키지 않음)(rescaler 만 해줌)

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
    batch_size=5,        # y 하나의 개수
    class_mode='binary'         # 이상이 있다-라벨 / 이상이 없다-라벨 : 이진분류
)
# Found 160 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    '../_data/brain/test',      # 이미지가 있는 폴더 지정이 아닌 상위 폴더(동일한 급의 라벨이 모여있는 폴더까지)로 지정   # train(ad/normal)
    target_size=(150, 150),     # 임의대로 크기 조정
    batch_size=5,        # y 하나의 개수
    class_mode='binary'         # 이상이 있다-라벨 / 이상이 없다-라벨 : 이진분류
)
# Found 120 images belonging to 2 classes.

# print(xy_train)
# <tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x000001CE349F8550>
print(xy_train[0])          # x, y
print(xy_train[0][0])       # x
print(xy_train[0][1])       # y
# print(xy_train[0][2])     # 없음
print(xy_train[0][0].shape, xy_train[0][1].shape)   # (5, 150, 150, 3) (5,)

print(xy_train[31][1]) # 마지막 배치 y => 이미지가 총 160장. 5로 나눴을 때 32가 되므로 
# print(xy_train[32][1]) 없다 

print(type(xy_train)) # <class 'tensorflow.python.keras.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0]))  # <class 'tuple'>
print(type(xy_train[0][0])) # <class 'numpy.ndarray'>
print(type(xy_train[0][1])) # <class 'numpy.ndarray'>

print(xy_test[23][1]) # 120장이기 때문에 마지막 배치가 된다 -> 배치 사이즈에 따라 달라진다 
# print(xy_test[24][1])
print(xy_test[0][0].shape, xy_test[0][1].shape) # (5, 150, 150, 3) (5,)


