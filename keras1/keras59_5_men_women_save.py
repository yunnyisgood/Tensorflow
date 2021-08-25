'''
실습 1
man women 데이터로 모델링을 구성할 것 
but, 용량이 너무 크기 때문에 문제 발생할 수도 
=> np.save, load 사용해서 용량 낮추기 

실습 2 <<< 과제 
본인 사진으로 predict 하시오

'''

from re import sub
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.python.keras.engine import training


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
    '../_data/men_women',      
    target_size=(80, 80),   
    batch_size=3310,        
    class_mode='binary',
    subset='training',
    shuffle=True         )
 # Found 3309 images belonging to 2 classes.   

xy_test = train_datagen.flow_from_directory(
    '../_data/men_women',      
    target_size=(80, 80),   
    batch_size=3310,        
    class_mode='binary',
    subset='validation',
    shuffle=True         )


print(type(xy_train))

print(xy_train[0][0].shape, xy_train[0][1].shape) 

print(xy_test[0][0].shape, xy_test[0][1].shape)


np.save('./_save/_npy/kerask59_men_women_x_train.npy', arr=xy_train[0][0])
np.save('./_save/_npy/kerask59_men_women_y_train.npy', arr=xy_train[0][1])
np.save('./_save/_npy/kerask59_men_women_x_test.npy', arr=xy_test[0][0])
np.save('./_save/_npy/kerask59_men_women_y_test.npy', arr=xy_test[0][1])