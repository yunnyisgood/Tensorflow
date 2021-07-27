import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, LSTM, Conv1D
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, QuantileTransformer, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import time

x_train = np.load('./_save/_npy/k55_x_train_cifar10.npy')
y_train = np.load('./_save/_npy/k55_y_train_cifar10.npy')
x_test = np.load('./_save/_npy/k55_x_test_cifar10.npy')
y_test = np.load('./_save/_npy/k55_y_test_cifar10.npy')

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# (50000, 32, 32, 3) (50000, 1) (10000, 32, 32, 3) (10000, 1)

# 데이터 전처러
x_train = x_train.reshape(50000, 32 * 32 * 3)
x_test = x_test.reshape(10000, 32 * 32 * 3)

# one-hot-encoding
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(50000, 32*3, 32)
x_test = x_test.reshape(10000, 32*3, 32)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# modeling -> RNN
# modeling
model = Sequential()
model.add(Conv2D(filters=100, kernel_size=(2, 2), 
                    padding='same', input_shape=(32, 32, 3), activation='relu'))
model.add(Conv2D(100, (2,2), activation='relu'))   
model.add(Conv2D(64, (2,2), activation='relu'))   
model.add(Conv2D(64, (2,2), activation='relu'))  
model.add(MaxPool2D()) 
model.add(Conv2D(32, (2,2), activation='relu')) 
model.add(Conv2D(32, (2,2), activation='relu')) 
model.add(MaxPool2D()) 
model.add(Flatten()) 
model.add(Dense(64, activation='relu'))         
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

# compile       -> metrics=['acc']
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', patience=1, verbose=1, mode='min')

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=50, verbose=1, callbacks=[es], validation_split=0.2,
shuffle=True, batch_size=256)
end_time = time.time() - start_time

# evaluate -> predict 할 필요는 없다
loss = model.evaluate(x_test, y_test)
print("걸린시간: ", end_time)
print('loss: ', loss[0])
print('accuracy: ', loss[1])