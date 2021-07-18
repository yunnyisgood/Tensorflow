import numpy as np
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

# 10개의 이미지를 분류하는 것
# 컬러 데이터

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape, y_train.shape)  # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)  # (10000, 32, 32, 3) (10000, 1)

print(np.unique(y_train)) # [0 1 2 3 4 5 6 7 8 9]

# 전처리 하기 -> one-hot-encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train.shape, y_test.shape)


# modeling
model = Sequential()
model.add(Conv2D(100, kernel_size=(2, 2), 
                    padding='same', input_shape=(32, 32, 3), activation='relu'))
model.add(Conv2D(100, (2,2), padding='same', activation='relu'))   
model.add(Conv2D(64, (2,2), padding='same', activation='relu'))   
model.add(Conv2D(64, (2,2),padding='same', activation='relu'))  
model.add(MaxPool2D()) 
model.add(Conv2D(32, (2,2), padding='same', activation='relu')) 
model.add(Conv2D(32, (2,2), padding='same', activation='relu')) 
model.add(MaxPool2D()) 
model.add(Flatten()) 
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(100, activation='softmax'))

# compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')

es = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')

model.fit(x_train, y_train, epochs=1000, verbose=1, callbacks=[es], validation_split=0.01,
shuffle=True, batch_size=200)

# evaluate 
loss = model.evaluate(x_test, y_test)
print('category: ', loss[0])
print('accuracy: ', loss[1])

# category:  2.8261590003967285
# accuracy:  0.31790000200271606