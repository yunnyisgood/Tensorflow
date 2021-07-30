import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
import time
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers.pooling import MaxPool1D
import matplotlib.pyplot as plt


x_train = np.load('./_save/_npy/kerask59_horse_x_train.npy')
y_train = np.load('./_save/_npy/kerask59_horse_y_train.npy')
x_test = np.load('./_save/_npy/kerask59_horse_x_test.npy')
y_test = np.load('./_save/_npy/kerask59_horse_y_test.npy')


print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# (771, 80, 80, 3) (771, 2) (256, 80, 80, 3) (256, 2)



train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=False,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.1,
    shear_range=0.5,
    fill_mode='nearest',
    )

augment_size=9229
# augment_size=154

randidx = np.random.randint(x_train.shape[0], size=augment_size)       # x_train[0]에서 아그먼트 사이즈 만큼 랜덤하게 들어감

print(x_train.shape[0])     # 60000
print(randidx)              # [44596 49164  1092 ... 51768  3501 13118]
print(randidx.shape)        # (40000,)

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()

print(x_augmented.shape)       # (40000, 28, 28)

import time
start = time.time()
                                
x_augmented = train_datagen.flow(x_augmented, np.zeros(augment_size), 
batch_size=augment_size, shuffle=False, 
# save_to_dir='d:/temp/'
).next()[0]

end = time.time() - start
print(x_augmented[0][0].shape)       
print(x_augmented[0][1].shape)  
print(x_augmented[0][1][:10])  
print(x_augmented[0][1][10:15])  
print('걸린시간 :', end)

x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))

print(x_train.shape, y_train.shape)
# (925, 80, 80, 3) (925, 2)


# modeling

# model = Sequential()
# model.add(Conv2D(filters = 8, kernel_size=(3,3), input_shape =(80,80,3), activation= 'relu'))
# # model.add(Dropout(0.2))
# model.add(Conv2D(filters = 8, kernel_size=(3,3), activation= 'relu'))
# model.add(MaxPooling2D(2,2))
# model.add(Conv2D(filters = 16, kernel_size=(2,2), activation= 'relu'))
# # model.add(Dropout(0.2))
# model.add(Conv2D(filters = 16, kernel_size=(2,2), activation= 'relu'))
# model.add(MaxPooling2D(2,2))
# model.add(Conv2D(filters = 32, kernel_size=(3,3), activation= 'relu'))

# model.add(Flatten())
# model.add(Dense(128, activation= 'relu'))
# model.add(Dense(64, activation= 'relu'))
# model.add(Dense(2, activation= 'softmax'))

model = Sequential()
model.add(Conv2D(128, kernel_size=(2, 2), 
                    padding='valid', input_shape=(32, 32, 3), activation='relu'))
# model.add(Dropout(0, 2)) # 20%의 드롭아웃의 효과를 낸다 
model.add(Dropout(0.2))
model.add(Conv2D(128, (2,2), padding='same', activation='relu'))   
model.add(MaxPooling2D()) 

# model.add(Conv2D(128, (2,2),padding='valid', activation='relu'))  
# model.add(Dropout(0.2))
# model.add(Conv2D(128, (2,2), padding='same', activation='relu')) 
# model.add(MaxPooling2D()) 

model.add(Conv2D(64, (2,2), padding='valid', activation='relu')) 
model.add(Dropout(0.2))
model.add(Conv2D(64, (2,2), padding='same', activation='relu')) 
model.add(MaxPooling2D()) 

model.add(GlobalAveragePooling2D())
model.add(Flatten())
model.add(Dense(128, activation= 'relu'))
model.add(Dense(64, activation= 'relu'))
model.add(Dense(2, activation='softmax'))

# compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='min', restore_best_weights=True)

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
Data 20% 증폭
loss:  0.7143795490264893
acc:  0.7256757020950317
val_acc:  0.5297297239303589

Data 10000개로 증폭 
loss:  0.6929205060005188
acc:  0.5235000252723694
val_acc:  0.5115000009536743
'''