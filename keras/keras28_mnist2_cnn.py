import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, QuantileTransformer, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical


# data

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,) -> 3차원
# print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

# 3차원 => 4차원으로 차원 수 늘려주기
# 단, 데이터의 내용물과 순서를 변경하면 안된다
# 차원의 계수만 같다면 전혀 상관 없다

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

print(np.unique(y_train)) 
# [0 1 2 3 4 5 6 7 8 9]
# -> 즉 값의 범위가 0~9까지라는 의미

# 전처리 하기 
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# scaler = MinMaxScaler()
# scaler.fit(x_train)
# scaler.transform(x_train)
# scaler.transform(x_test)


# modeling
model = Sequential()
model.add(Conv2D(filters=100, kernel_size=(2, 2), 
                    padding='same', input_shape=(28, 28, 1)))
model.add(Conv2D(100, (2,2), activation='relu'))   
model.add(Conv2D(64, (2,2), activation='relu'))   
model.add(Conv2D(64, (2,2), activation='relu'))  
model.add(MaxPooling2D()) 
model.add(Conv2D(32, (2,2), activation='relu')) 
model.add(Conv2D(32, (2,2), activation='relu')) 
model.add(MaxPooling2D()) 
model.add(Flatten()) 
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

# compile       -> metrics=['acc']
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='min')

model.fit(x_train, y_train, epochs=1000, verbose=1, callbacks=[es], validation_split=0.01,
shuffle=True)

# evaluate -> predict 할 필요는 없다
loss = model.evaluate(x_test, y_test)
print('loss: ', loss[0])
print('accuracy: ', loss[1])


# loss:  0.46171167492866516
# accuracy:  0.980400025844574