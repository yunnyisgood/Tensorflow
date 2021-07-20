import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, GlobalAveragePooling1D, LSTM
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, QuantileTransformer, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import time

# data

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train.shape, y_train.shape)  # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

x_train = x_train.reshape(60000, 28* 28* 1) # (60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28* 28* 1) # (10000, 28, 28, 1)

print(np.unique(y_train)) # [0 1 2 3 4 5 6 7 8 9]

# 전처리 하기: 3차원 -> 2차원 
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 다시 3차원으로 

x_train = x_train.reshape(60000, 28 , 28)
x_test = x_test.reshape(10000, 28 , 28)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# modeling
'''model = Sequential()
model.add(Conv2D(filters=100, kernel_size=(2, 2), 
                    padding='same', input_shape=(28, 28, 1)))
model.add(Conv2D(100, (2,2), padding='same', activation='relu'))   
model.add(Conv2D(64, (2,2), padding='same', activation='relu'))   
model.add(Conv2D(64, (2,2), padding='same', activation='relu'))  
# model.add(MaxPooling2D()) -> 왜 이렇게 하면 loss가 안나올까?
model.add(Conv2D(32, (2,2), activation='relu')) 
model.add(Conv2D(32, (2,2), activation='relu')) 
model.add(MaxPooling2D()) 
model.add(Flatten()) 
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

# modeling -> DNN
model = Sequential()
# model.add(Dense(5000, input_shape=(28* 28, ), activation='relu'))
model.add(Dense(5000, input_shape=(28* 28, 1), activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dense(10, activation='softmax')) '''

# modeling -> RNN
model = Sequential()
model.add(LSTM(units=32, activation='relu', input_shape=(28,28)))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(10))

model.summary()

# compile       -> metrics=['acc']
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='min')

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=1000, verbose=1, callbacks=[es], validation_split=0.2,
shuffle=True, batch_size=256)
end_time = time.time() - start_time

# evaluate -> predict 할 필요는 없다
loss = model.evaluate(x_test, y_test)
print("걸린시간: ", end_time)
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


loss:  0.7816981673240662
accuracy:  0.9120000004768372

DNN 으로 돌렸을 떄 
걸린시간:  29.480650424957275
loss:  0.562569797039032
accuracy:  0.892799973487854

DNN + GPA 0 -> 차원 늘려줬을 떄 
-> 메모리 뻥~

RNN으로 돌렸을 때  
걸린시간:  441.56017780303955
loss:  9.24865436553955
accuracy:  0.14839999377727509

'''