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
from keras.preprocessing import image
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, QuantileTransformer, StandardScaler
from tensorflow.keras.applications import VGG16, VGG19



x_train = np.load('./_save/kerask59_men_women_x_train.npy')
y_train = np.load('./_save/kerask59_men_women_y_train.npy')
x_test = np.load('./_save/kerask59_men_women_x_test.npy')
y_test = np.load('./_save/kerask59_men_women_y_test.npy')
x_pred = np.load('./_save/kerask59_men_women_mine2.npy')

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape, x_pred.shape)
# (2486, 80, 80, 3) (2483,) (828, 80, 80, 3) (827,) (5, 80, 80, 3)


x_train = x_train.reshape(2486, 80 * 80*3)
x_test = x_test.reshape(828, 80 * 80*3)

print(x_train.shape, x_test.shape) 

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(2486, 80 , 80, 3)
x_test = x_test.reshape(828, 80, 80, 3)

vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(80, 80, 3))

# vgg16.trainable =True  # vgg훈련을 동결한다 -> 0이 된다 
vgg16.trainable =True  # vgg훈련을 동결한다 -> 0이 된다 

model = Sequential()
model.add(vgg16)
# model.add(GlobalAveragePooling2D())
model.add(Flatten())
model.add(Dense(128, activation= 'relu'))
model.add(Dense(64, activation= 'relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.trainable = True

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
start_time = time.time()
es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=1)
model.fit(x_train, y_train, verbose=1, epochs=100, batch_size=256, validation_split=0.2,callbacks=es)#  callbacks=es)

# evaluate 
loss = model.evaluate(x_test, y_test)
print('loss: ', loss[0])
print('accuracy: ', loss[1])
print("걸린 시간: ", time.time()-start_time)

# 내 사진으로 예측하기 
y_pred = model.predict([x_pred])
print('y_pred 1:', y_pred)
# [[1.][1.][1.][1.][1.]]

y_pred = y_pred * 100

print('여자일 확률 : ', y_pred[0], '%')
# 여자일 확률 : 100%

'''
>> vgg16
> both True
GAP
loss:  0.0
accuracy:  0.001207729452289641
걸린 시간:  155.31352972984314
여자일 확률 :  [100.] %

Flatten
loss:  0.0
accuracy:  0.001207729452289641
걸린 시간:  154.89455485343933

'''