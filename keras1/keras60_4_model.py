from numpy.core.fromnumeric import size
from tensorflow.keras.datasets import fashion_mnist
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, GlobalAveragePooling1D, LSTM, Conv1D
import time
from tensorflow.keras.callbacks import EarlyStopping

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
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest'  
    )

# xy_train = train_datagen.flow_from_directory(
#     '../_data/cat_and_dog/training_set/training_set',     
#     target_size=(32, 32),     
#     batch_size=8100,       
#     class_mode='binary'  )     

augmented_size = 40000

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

x_augmented = train_datagen.flow(x_augmented, np.zeros(augmented_size),
                                    batch_size=augmented_size, shuffle=False).next()[0]
# 1개의 데이터가 4만개로 증폭되는 것이 아니라, 4만개의 데이터가                                     
print(x_augmented[0][0].shape) # (40000, 28, 28, 1)
print(x_augmented[0][1].shape) # (40000,)
print(x_augmented.shape) # NumpyArrayIterator이기 때문에 출력 불가 
print(x_augmented[0].shape) # tuple이라 출력불가 

# next() 후에 
print(x_augmented[0][0].shape) # (28, 1)
print(x_augmented[0][1].shape) # (28, 1)
print(x_augmented.shape) # (40000, 28, 28, 1)
print(x_augmented[0].shape) # (28, 28, 1)

x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))

print(x_train.shape, y_train.shape) # (100000, 28, 28, 1) (100000,)
print(x_test.shape, y_test.shape) # (10000, 28, 28, 1) (10000,)

'''
모델 완성
비교대싱? 
데이터 증폭이 과연 과적합을 해결했는가?
기존의 fashion mnist와 비교 
'''

# modeling
# model = Sequential()
# model.add(Conv2D(filters=100, kernel_size=(2, 2), 
#                     padding='same', input_shape=(28, 28, 1)))
# model.add(Conv2D(100, (2,2), padding='same', activation='relu'))   
# model.add(Conv2D(64, (2,2), padding='same', activation='relu'))   
# model.add(Conv2D(64, (2,2), padding='same', activation='relu'))  
# # model.add(MaxPooling2D()) -> 왜 이렇게 하면 loss가 안나올까?
# model.add(Conv2D(32, (2,2), activation='relu')) 
# model.add(Conv2D(32, (2,2), activation='relu')) 
# model.add(MaxPooling2D()) 
# model.add(Flatten()) 
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(10, activation='softmax'))

model = Sequential()
model.add(Conv2D(filters = 8, kernel_size=(3,3), input_shape =(28,28,1), activation= 'relu'))
# model.add(Dropout(0.2))
model.add(Conv2D(filters = 8, kernel_size=(3,3), activation= 'relu'))
model.add(MaxPooling2D(2,2))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(filters = 16, kernel_size=(2,2), activation= 'relu'))
# model.add(Dropout(0.2))
model.add(Conv2D(filters = 16, kernel_size=(2,2), activation= 'relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(filters = 32, kernel_size=(3,3), activation= 'relu'))

model.add(Flatten())
model.add(Dense(128, activation= 'relu'))
model.add(Dense(64, activation= 'relu'))
model.add(Dense(10, activation= 'softmax'))


# compile
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min', restore_best_weights=True)

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=1000, verbose=1, callbacks=[es], validation_split=0.2,
shuffle=True, batch_size=9)

end_time = time.time() - start_time

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']


# evaluate 
loss = model.evaluate(x_test, y_test)
print('loss: ', loss[0])
print('acc: ', acc[-1])
print('val_acc: ', val_acc[-1])


# 시각화 
plt.figure(figsize=(9,5))

# 1
plt.subplot(2, 1, 1) 
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

# 2
plt.subplot(2, 1, 2) 
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.grid()
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['acc', 'val_acc'])

plt.show()

'''
loss:  0.3244667053222656
acc:  0.829675018787384
val_acc:  0.5516499876976013

modeling 변경 후 

'''