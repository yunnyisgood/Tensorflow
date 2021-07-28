'''
실습 1
man women 데이터로 모델링을 구성할 것 
but, 용량이 너무 크기 때문에 문제 발생할 수도 
=> np.save, load 사용해서 용량 낮추기 

실습 2 <<< 과제 
본인 사진으로 predict 하시오

'''

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
import time
from sklearn.model_selection import train_test_split
from keras.preprocessing import image


x_train = np.load('./_save/_npy/kerask59_men_women_x_train.npy')
y_train = np.load('./_save/_npy/kerask59_men_women_y_train.npy')
x_test = np.load('./_save/_npy/kerask59_men_women_x_test.npy')
y_test = np.load('./_save/_npy/kerask59_men_women_y_test.npy')
x_pred = np.load('./_save/_npy/kerask59_men_women_mine2.npy')

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape, x_pred.shape)
# (2483, 80, 80, 3) (2483,) (827, 80, 80, 3) (827,) (5, 80, 80, 3)


# img = image.load_img('../_data/men_women/mine/mine.jpeg', target_size=(64, 64))
# x_pred = image.img_to_array(img)
# x_pred = np.expand_dims(x_pred, axis=0)
# x_pred = np.vstack([x_pred])


# modling

model = Sequential()
model.add(Conv2D(filters = 8, kernel_size=(3,3), input_shape =(80,80,3), activation= 'relu'))
# model.add(Dropout(0.2))
model.add(Conv2D(filters = 8, kernel_size=(3,3), activation= 'relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(filters = 16, kernel_size=(2,2), activation= 'relu'))
# model.add(Dropout(0.2))
model.add(Conv2D(filters = 16, kernel_size=(2,2), activation= 'relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(filters = 32, kernel_size=(3,3), activation= 'relu'))

model.add(Flatten())
model.add(Dense(128, activation= 'relu'))
model.add(Dense(64, activation= 'relu'))
model.add(Dense(1, activation= 'sigmoid'))


# compile
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

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

# 내 사진으로 예측하기 
y_pred = model.predict([x_pred])
print('y_pred 1:', y_pred)
# [[1.][1.][1.][1.][1.]]

y_pred = y_pred * 100

print('여자일 확률 : ', y_pred[0], '%')
# 여자일 확률 : 100%


'''
loss:  -3.8648104635991484e+20
acc:  0.431017130613327
val_acc:  0.40442654490470886
여자일 확률 :  [100.] %
'''

