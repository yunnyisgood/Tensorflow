import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, LSTM, Conv1D
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, QuantileTransformer, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import time

from tensorflow.python.keras.saving.save import load_model

# 10개의 이미지를 분류하는 것
# 컬러 데이터

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape)  # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)  # (10000, 32, 32, 3) (10000, 1)

print(np.unique(y_train)) # [0 1 2 3 4 5 6 7 8 9]

x_train = x_train.reshape(50000, 32 * 32 * 3)
x_test = x_test.reshape(10000, 32 * 32 * 3)

# 전처리 하기 -> one-hot-encoding
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(50000, 32*3, 32)
x_test = x_test.reshape(10000, 32*3, 32)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# modeling
'''model = Sequential()
model.add(Conv2D(filters=100, kernel_size=(2, 2), 
                    padding='same', input_shape=(32*32*3, ), activation='relu'))
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


# modeling -> DNN
model = Sequential()
model.add(Dense(5000, input_shape=(32*32*3, ), activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax')) '''

# modeling -> RNN
'''model = Sequential()
model.add(LSTM(units=32, activation='relu', input_shape=(32*3,32),return_sequences=True))
model.add(Conv1D(10, 2, activation='relu'))
model.add(Conv1D(10, 2, activation='relu'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

# compile       -> metrics=['acc']
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', patience=1, verbose=1, mode='min')
cp = ModelCheckpoint(monitor='val_loss', save_best_only=True, mode='auto', 
filepath='./_save/ModelCheckPoint/keras47_MCP_cifar10.hdf5')

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=50, verbose=1, callbacks=[es, cp], validation_split=0.2,
shuffle=True, batch_size=256)
end_time = time.time() - start_time

model.save('./_save/ModelCheckPoint/keras47_MCP_cifar10.h5')'''

model = load_model('./_save/ModelCheckPoint/keras47_MCP_cifar10.hdf5')

# evaluate -> predict 할 필요는 없다
loss = model.evaluate(x_test, y_test)
# print("걸린시간: ", end_time)
print('loss: ', loss[0])
print('accuracy: ', loss[1])


'''
# 시각화  
plt.figure(figsize=(9,5))

# 1
plt.subplot(2, 1, 1) # 2개의 플롯을 할건데, 1행 1열을 사용하겠다는 의미 
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

# 2
plt.subplot(2, 1, 2) # 2개의 플롯을 할건데, 1행 2열을 사용하겠다는 의미 
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.grid()
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['acc', 'val_acc'])

plt.show()



loss:  1.2721869945526123
accuracy:  0.6816999912261963

DNN 으로 도출
> Standard Scaler
걸린시간:  41.20919370651245
loss:  2.4278764724731445
accuracy:  0.5199999809265137

> MinMax Sclaer
걸린시간:  66.00103783607483
loss:  1.5630486011505127
accuracy:  0.5184000134468079

RNN으로 도출 
걸린시간:  407.3573589324951
loss:  8.381452560424805
accuracy:  0.010400000028312206

Conv1D
걸린시간:  252.61560988426208
loss:  1.7440955638885498
accuracy:  0.37610000371932983

MCP [전] -> early stopping 지점
걸린시간:  377.0775468349457
loss:  1.5050286054611206
accuracy:  0.45170000195503235

MCP [후] -> modelCheckPoint 지점 
loss:  1.4755481481552124
accuracy:  0.460999995470047

'''