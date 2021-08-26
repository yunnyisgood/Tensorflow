import numpy as np
from tensorflow.keras.datasets import cifar100, mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, PowerTransformer, QuantileTransformer
import time 
import matplotlib.pyplot as plt

'''
# 10개의 이미지를 분류하는 것
# 컬러 데이터

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,) -> 3차원
# print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

print(np.unique(y_train)) 

x_train = x_train.reshape(60000, 28*28)
x_test = x_test.reshape(10000,  28*28)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000,  28, 28, 1)


# 전처리 하기 -> one-hot-encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train.shape, y_test.shape)
'''

# modeling
model = Sequential()
model.add(Conv2D(128, kernel_size=(2, 2), 
                    padding='valid', input_shape=(28, 28, 1), activation='relu'))
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

model.add(Flatten()) 
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

model.save('./_save/keras45_1_save_model.h5')

'''
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

'''

'''
dropout 실행
걸린시간:  603.1085088253021
category:  2.0480213165283203
accuracy:  0.4699999988079071

Conv1D
걸린시간:  221.4401044845581
category:  0.025295158848166466
accuracy:  0.9943000078201294

'''