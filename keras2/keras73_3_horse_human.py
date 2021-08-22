'''
가장 잘 나온 전이학습 모델로
이 데이터를 학습시켜서 결과치 도출
keras 59와의 성능 비교
'''
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
import time
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers.pooling import MaxPool1D
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16, VGG19


x_train = np.load('./_save/_npy/kerask59_horse_x_train.npy')
y_train = np.load('./_save/_npy/kerask59_horse_y_train.npy')
x_test = np.load('./_save/_npy/kerask59_horse_x_test.npy')
y_test = np.load('./_save/_npy/kerask59_horse_y_test.npy')


print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(80, 80, 3))

# vgg16.trainable =True  # vgg훈련을 동결한다 -> 0이 된다 
vgg16.trainable =True  # vgg훈련을 동결한다 -> 0이 된다 

model = Sequential()
model.add(vgg16)
# model.add(GlobalAveragePooling2D())
model.add(Flatten())
model.add(Dense(128, activation= 'relu'))
model.add(Dense(64, activation= 'relu'))
model.add(Dense(2, activation='softmax'))

model.summary()

model.trainable = True

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
start_time = time.time()
es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=1)
model.fit(x_train, y_train, verbose=1, epochs=100, batch_size=512, validation_split=0.2,callbacks=es)#  callbacks=es)

# evaluate 
loss = model.evaluate(x_test, y_test)
print('loss: ', loss[0])
print('accuracy: ', loss[1])
print("걸린 시간: ", time.time()-start_time)