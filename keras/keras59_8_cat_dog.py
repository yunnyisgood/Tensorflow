import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


'''
sigmoid와 categorical_crossentropy 조합으로 실행
'''


train_datagen = ImageDataGenerator(
    rescale=1./255,  
    horizontal_flip=True, 
    vertical_flip=True,
    width_shift_range=0.1, # 
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest'  
    )

test_datagen = ImageDataGenerator(rescale=1./255)

xy_train = train_datagen.flow_from_directory(
    '../_data/cat_and_dog/training_set/training_set',     
    target_size=(32, 32),     
    batch_size=8100,       
    class_mode='binary'  )    
# Found 8005 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    '../_data/cat_and_dog/test_set/test_set',    
    target_size=(32, 32),     
    batch_size=2030,       
    class_mode='binary'         
)
# Found 2023 images belonging to 2 classes.

print(xy_train[0].shape, xy_train[1].shape)  
# (8005, 32, 32, 3) (8005,)

print(xy_train[0][0].shape, xy_train[0][1].shape)  
# (8005, 80, 80, 3) (8005,)
# .next() -> (8005, 32, 32, 3) (8005,)
print(xy_test[0][0].shape, xy_test[0][1].shape) 
# (2023, 80, 80, 3) (2023,)
# .next() -> (8005, 32, 32, 3) (8005,)

# np.save('./_save/_npy/kerask59_8_train_x32.npy', arr=xy_train[0][0])
# np.save('./_save/_npy/kerask59_8_train_y32.npy', arr=xy_train[0][1])
# np.save('./_save/_npy/kerask59_8_test_x32.npy', arr=xy_test[0][0])
# np.save('./_save/_npy/kerask59_8_test_y32.npy', arr=xy_test[0][1])