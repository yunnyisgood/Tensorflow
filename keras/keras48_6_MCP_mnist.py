import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Reshape, Conv1D
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, QuantileTransformer, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.saving.save import load_model


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
'''model = Sequential()
model.add(Dense(units=10, activation='relu', input_shape=(28,28)))
model.add(Conv1D(10, 2, activation='relu'))
model.add(Conv1D(10, 2, activation='relu'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(10, activation='softmax'))

# compile       -> metrics=['acc']
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')

cp = ModelCheckpoint(monitor='val_loss', save_best_only=True, mode='auto', 
filepath='./_save/ModelCheckPoint/keras47_MCP_mnist.hdf5')

model.fit(x_train, y_train, epochs=100, verbose=1, callbacks=[es, cp], validation_split=0.01,
shuffle=True, batch_size=256)

model.save('./_save/ModelCheckPoint/keras47_MCP_mnist.h5')'''
model = load_model('./_save/ModelCheckPoint/keras47_MCP_mnist.hdf5')

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

Conv1D
loss:  0.18722544610500336
accuracy:  0.9501000046730042

MCP [전] -> early stopping 지점
loss:  0.17128945887088776
accuracy:  0.9598000049591064

MCP [후] -> modelCheckPoint 지점 
loss:  0.18074704706668854
accuracy:  0.9567000269889832
'''  