import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
import time
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers.pooling import MaxPool1D
import matplotlib.pyplot as plt


x_train = np.load('./_save/kerask59_rps_x_train.npy')
y_train = np.load('./_save/kerask59_rps_y_train.npy')
x_test = np.load('./_save/kerask59_rps_x_test.npy')
y_test = np.load('./_save/kerask59_rps_y_test.npy')



print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# (1890, 80, 80, 3) (1890, 3) (630, 80, 80, 3) (630, 3)

model = Sequential()

model.add(Conv2D(filters = 8, kernel_size=(3,3), input_shape =(80,80,3), activation= 'relu'))
model.add(Dropout(0.2))
model.add(Conv2D(filters = 8, kernel_size=(3,3), activation= 'relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(filters = 16, kernel_size=(2,2), activation= 'relu'))
model.add(Dropout(0.2))
model.add(Conv2D(filters = 16, kernel_size=(2,2), activation= 'relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(filters = 32, kernel_size=(3,3), activation= 'relu'))

model.add(Flatten())
model.add(Dense(128, activation= 'relu'))
model.add(Dense(64, activation= 'relu'))
model.add(Dense(3, activation= 'softmax'))


# compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min', restore_best_weights=True)

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=100, verbose=1, callbacks=[es], validation_split=0.2,
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
loss:  2.4712209701538086
acc:  0.955320417881012
val_acc:  0.6084656119346619

batch_size=8
loss:  1.0986186265945435
acc:  0.31283068656921387
val_acc:  0.33068782091140747




'''

