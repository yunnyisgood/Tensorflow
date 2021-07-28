import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten


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
    validation_split=0.25
    )

test_datagen = ImageDataGenerator(rescale=1./255)

xy_train = train_datagen.flow_from_directory(
    '../_data/rps',      
    target_size=(80, 80),   
    batch_size=2530,        
    class_mode='categorical',
    shuffle=True,
    subset='training'         )
# Found 1890 images belonging to 3 classes.

xy_test = train_datagen.flow_from_directory(
    '../_data/rps',     
    target_size=(80, 80),   
    batch_size=2530,       
    class_mode='categorical',
    shuffle=True,
    subset='validation'  
)
# Found 630 images belonging to 3 classes.

print(type(xy_train))

print(xy_train[0][0].shape, xy_train[0][1].shape) 
# (1890, 80, 80, 3) (1890, 3)
print(xy_test[0][0].shape, xy_test[0][1].shape) 
# (630, 80, 80, 3) (630, 3)

np.save('./_save/_npy/kerask59_rps_x_train.npy', arr=xy_train[0][0])
np.save('./_save/_npy/kerask59_rps_y_train.npy', arr=xy_train[0][1])
np.save('./_save/_npy/kerask59_rps_x_test.npy', arr=xy_test[0][0])
np.save('./_save/_npy/kerask59_rps_y_test.npy', arr=xy_test[0][1])