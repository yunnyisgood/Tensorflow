import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, PowerTransformer, QuantileTransformer
import time 

x_train = np.load('./_save/_npy/k55_x_train_cifar100.npy')
y_train = np.load('./_save/_npy/k55_y_train_cifar100.npy')
x_test = np.load('./_save/_npy/k55_x_test_cifar100.npy')
y_test = np.load('./_save/_npy/k55_y_test_cifar100.npy')

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# (50000, 32, 32, 3) (50000, 1) (10000, 32, 32, 3) (10000, 1)


# modeling
model = Sequential()
model.add(Conv2D(128, kernel_size=(2, 2), 
                    padding='valid', input_shape=(32, 32, 3), activation='relu'))
# model.add(Dropout(0, 2)) # 20%의 드롭아웃의 효과를 낸다 
model.add(Dropout(0.2))
model.add(Conv2D(128, (2,2), padding='same', activation='relu'))   
model.add(MaxPool2D()) 

model.add(Conv2D(128, (2,2),padding='valid', activation='relu'))  
model.add(Dropout(0.2))
model.add(Conv2D(128, (2,2), padding='same', activation='relu')) 
model.add(MaxPool2D()) 

model.add(Conv2D(64, (2,2), padding='valid', activation='relu')) 
model.add(Dropout(0.2))
model.add(Conv2D(64, (2,2), padding='same', activation='relu')) 
model.add(MaxPool2D()) 

model.add(GlobalAveragePooling2D())
model.add(Dense(100, activation='softmax'))



# compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=100, verbose=1, callbacks=[es], validation_split=0.2,
shuffle=True, batch_size=256)
end_time = time.time() - start_time



# evaluate 
loss = model.evaluate(x_test, y_test)
print("걸린시간: ", end_time)
print('category: ', loss[0])
print('accuracy: ', loss[1])

