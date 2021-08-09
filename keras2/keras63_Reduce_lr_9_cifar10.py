import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, QuantileTransformer, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import time

# 10개의 이미지를 분류하는 것
# 컬러 데이터

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape)  # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)  # (10000, 32, 32, 3) (10000, 1)

print(np.unique(y_train)) # [0 1 2 3 4 5 6 7 8 9]

# 전처리 하기 -> one-hot-encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train.shape, y_test.shape)


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

# 3. compile
optimizer = Adam(lr=0.001)

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

es = EarlyStopping(monitor='val_loss', patience=8, verbose=1, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto', verbose=1, factor=0.5)

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=100, verbose=1, callbacks=[es, reduce_lr], validation_split=0.2,
shuffle=True)
end_time = time.time() - start_time

# evaluate 
loss = model.evaluate(x_test, y_test)
print('loss: ', loss[0])
print('accuracy: ', loss[1])

# loss:  1.2721869945526123
# accuracy:  0.6816999912261963

# Adam(lr=0.001)
# loss:  1.155794382095337
# accuracy:  0.686900019645690