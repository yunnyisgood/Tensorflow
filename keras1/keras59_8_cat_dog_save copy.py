import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


'''
sigmoid와 categorical_crossentropy 조합으로 실행
'''


## ImageDataGenerator로 데이터 증폭시키기
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest',
    validation_split=0.2
    )
test_datagen = ImageDataGenerator(rescale=1./255)




trainGen = train_datagen.flow_from_directory(
    '../_data/cat_and_dog/training_set/training_set',     
    target_size=(150, 150),
    batch_size=2000,
    class_mode='categorical'
)
# Found 8005 images belonging to 2 classes.
print(trainGen[0][0].shape)     # (2000, 150, 150, 3)
print(trainGen[0][1].shape)     # (2000, 1)


testGen = test_datagen.flow_from_directory(
    '../_data/cat_and_dog/test_set/test_set',  
    target_size=(150, 150),
    batch_size=1000,
    class_mode='categorical',
)
# FFound 2023 images belonging to 2 classes.
print(testGen[0][0].shape)     # (1000, 150, 150, 3)
print(testGen[0][1].shape)     # (1000, 2)


np.save('./_save/kerask59_8_train_x64.npy', arr=trainGen[0][0])
np.save('./_save/kerask59_8_train_y64.npy', arr=trainGen[0][1])
np.save('./_save/kerask59_8_test_x64.npy', arr=testGen[0][0])
np.save('./_save/kerask59_8_test_y64.npy', arr=testGen[0][1])