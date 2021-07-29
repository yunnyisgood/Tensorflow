from tensorflow.keras.datasets import fashion_mnist
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.backend import zeros

'''
ImageDataGenerator를 사용한 
데이터 증폭 
'''

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

train_datagen = ImageDataGenerator(
    rescale=1./255,  
    horizontal_flip=True, 
    # vertical_flip=True,
    vertical_flip=False,
    width_shift_range=0.1, # 
    height_shift_range=0.1,
    rotation_range=5,
    # zoom_range=1.2,
    zoom_range=0.1,
    # shear_range=0.7,
    shear_range=0.5,
    fill_mode='nearest'  
    )

# xy_train = train_datagen.flow_from_directory(
#     '../_data/cat_and_dog/training_set/training_set',     
#     target_size=(32, 32),     
#     batch_size=8100,       
#     class_mode='binary'  )     

augmented_size = 100
x_data = train_datagen.flow(
    np.tile(x_train[0].reshape(28*28), augmented_size).reshape(-1, 28, 28, 1), # x값 100개로 증폭 
    np.zeros(augmented_size), # y값은 0으로 임의로 채워준다 
    batch_size=augmented_size,
    shuffle=False

).next()    

print(type(x_data))
print(type(x_data[0])) # <class 'tuple'> -> 여러개의 array가 모여 하나의 튜플로 이루어지게 된다 
print(type(x_data[0][0])) # <class 'numpy.ndarray'>
print(x_data[0][0].shape) # (100, 28, 28, 1) -> y값 없이 x값만 분리되어 출력된다 
print(x_data[0][1].shape) # (100,)


# .next()
print(x_data[0].shape) #(100, 28, 28, 1)
print(x_data[1].shape) # (100,)
print(x_data[0][0].shape) # (28, 28, 1)
print(x_data[0][1].shape) # (28, 28, 1)



import matplotlib.pyplot as plt
plt.figure(figsize=(7, 7))
for i in range(49):
    plt.subplot(7, 7, i+1)
    plt.axis('off')
    plt.imshow(x_data[0][i], cmap='gray')
plt.show()
