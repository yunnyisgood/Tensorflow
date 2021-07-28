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
    '../_data/horse-or-human',      
    target_size=(80, 80),   
    batch_size=1030,        
    class_mode='categorical',
    subset='training',
    shuffle=True         )

# Found 1027 images belonging to 2 classes.    

xy_test = train_datagen.flow_from_directory(
    '../_data/horse-or-human',      
    target_size=(80, 80),   
    batch_size=1030,        
    class_mode='categorical',
    subset='validation',
    shuffle=True          )    


# xy_test = test_datagen.flow_from_directory(
#     '../_data/men_women/mine',     
#     target_size=(32, 32),   
#     batch_size=100,       
#     class_mode='binary'        
# )


print(type(xy_train))

print(xy_train[0][0].shape, xy_train[0][1].shape) 
# (822, 64, 64, 3) (822, 2)
# (771, 64, 64, 3) (771, 2)
print(xy_test[0][0].shape, xy_test[0][1].shape)
# (205, 64, 64, 3) (205, 2)
# (256, 64, 64, 3) (256, 

np.save('./_save/_npy/kerask59_horse_x_train.npy', arr=xy_train[0][0])
np.save('./_save/_npy/kerask59_horse_y_train.npy', arr=xy_train[0][1])
np.save('./_save/_npy/kerask59_horse_x_test.npy', arr=xy_test[0][0])
np.save('./_save/_npy/kerask59_horse_y_test.npy', arr=xy_test[0][1])