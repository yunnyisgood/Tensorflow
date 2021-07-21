import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Reshape
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, QuantileTransformer, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical


# data

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,) -> 3차원
# print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

print(np.unique(y_train)) 
# [0 1 2 3 4 5 6 7 8 9]
# -> 즉 값의 범위가 0~9까지라는 의미

# 전처리 하기 
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# modeling
model = Sequential()
model.add(Dense(units=10, activation='relu', input_shape=(28,28)))
model.add(Flatten()) # (None, 280)
model.add(Dense(64, activation='relu')) # (None, 64)
model.add(Reshape((8, 8, 1))) # (None, 64) -> (None, 8, 8, 1)
model.add(Conv2D(64, (2,2),padding='same', activation='relu'))
model.add(Conv2D(64, (2,2),padding='same', activation='relu'))
model.add(Conv2D(32, (2,2),padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten()) 
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation='softmax'))

# compile       -> metrics=['acc']
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='min')

model.fit(x_train, y_train, epochs=1000, verbose=1, callbacks=[es], validation_split=0.01,
shuffle=True, batch_size=256)

# evaluate -> predict 할 필요는 없다
loss = model.evaluate(x_test, y_test)
print('loss: ', loss[0])
print('accuracy: ', loss[1])


''' 
loss:  0.46171167492866516
accuracy:  0.980400025844574

DNN 2차원으로 받아서 
loss:  0.15973235666751862
accuracy:  0.9682000279426575

N차원으로 받아서 flatten
loss:  0.15187221765518188
accuracy:  0.97079998254776

DNN + CNN
loss:  0.4026147425174713
accuracy:  0.9341999888420105
'''  