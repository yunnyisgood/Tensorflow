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
    )

test_datagen = ImageDataGenerator(rescale=1./255)


x_pred = train_datagen.flow_from_directory(
    '../_data/men_women/mine',     
    target_size=(80, 80),   
    batch_size=3310,       
    class_mode='binary',
              
)


print(type(x_pred))
print(x_pred[0][0])


np.save('./_save/_npy/kerask59_men_women_mine2.npy', arr=x_pred[0][0])

