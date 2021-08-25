import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, QuantileTransformer, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

x_train = np.load('./_save/_npy/k55_x_train_mnist.npy')
y_train = np.load('./_save/_npy/k55_y_train_mnist.npy')
x_test = np.load('./_save/_npy/k55_x_test_mnist.npy')
y_test = np.load('./_save/_npy/k55_y_test_mnist.npy')


print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# (60000, 28, 28) (60000,) (10000, 28, 28) (10000,)


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