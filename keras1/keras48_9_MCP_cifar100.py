import numpy as np
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, LSTM, Conv1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, PowerTransformer, QuantileTransformer
import time 
import matplotlib.pyplot as plt
from tensorflow.python.keras.saving.save import load_model


# 10개의 이미지를 분류하는 것
# 컬러 데이터

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

x_train = x_train.reshape(50000, 32 * 32 * 3)
x_test = x_test.reshape(10000, 32 * 32 * 3)

print(np.unique(y_train)) 

# 전처리 하기 -> scailing
# 단, 2차원 데이터만 가능하므로 4차원 -> 2차원
# x_train = x_train/255.
# x_test = x_test/255.

print(x_train.shape, x_test.shape) # (50000, 3072) (10000, 3072)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(50000, 32*3, 32)
x_test = x_test.reshape(10000, 32*3, 32)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train.shape, y_test.shape)

'''
# modeling -> DNN
model = Sequential()
model.add(Dense(5000, input_shape=(32*32*3, ), activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(100, activation='softmax')) '''

# modeling -> RNN
'''model = Sequential()
model.add(LSTM(units=32, activation='relu', input_shape=(32*3, 32), return_sequences=True))
model.add(Conv1D(10, 2, activation='relu'))
model.add(Conv1D(10, 2, activation='relu'))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(100))

model.summary()

# compile       -> metrics=['acc']
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', patience=1, verbose=1, mode='min')
cp = ModelCheckpoint(monitor='val_loss', save_best_only=True, mode='auto', 
filepath='./_save/ModelCheckPoint/keras47_MCP_cifar100.hdf5')

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=50, verbose=1, callbacks=[es, cp], validation_split=0.2,
shuffle=True, batch_size=256)
end_time = time.time() - start_time

model.save('./_save/ModelCheckPoint/keras47_MCP_cifar100.h5')'''

model = load_model('./_save/ModelCheckPoint/keras47_MCP_cifar100.hdf5')

# evaluate -> predict 할 필요는 없다
loss = model.evaluate(x_test, y_test)
# print("걸린시간: ", end_time)
print('loss: ', loss[0])
print('accuracy: ', loss[1])

'''
category:  2.8261590003967285
accuracy:  0.31790000200271606

MinMax
category:  3.1839680671691895
accuracy:  0.33649998903274536

Standard
걸린시간:  402.19969177246094
category:  3.1498517990112305
accuracy:  0.35659998655319214


patience, batch 줄이고 validation 늘렸을 때 
category:  3.038492441177368
accuracy:  0.3449999988079071

validation 높이고, modeling 수정
걸린시간:  174.78156971931458
category:  3.2364041805267334
accuracy:  0.37290000915527344

batch_size 더 줄였을때 128-> 64
걸린시간:  207.46179294586182
category:  3.2013678550720215
accuracy:  0.3716000020503998

batch_size 64 -> 256 늘렸을 떄
걸린시간:  151.7369945049286
category:  2.806745767593384
accuracy:  0.3878999948501587

DNN으로 실행했을 때 
걸린시간:  68.33044385910034
loss:  3.6031110286712646
accuracy:  0.2110999971628189

RNN으로 실행했을 때 
걸린시간:  377.16542649269104
loss:  8.23963737487793
accuracy:  0.008999999612569809

Conv1D
걸린시간:  159.36642169952393
loss:  7.973621845245361
accuracy:  0.009999999776482582

MCP [전] -> early stopping 지점
걸린시간:  97.16229701042175
loss:  10.154399871826172
accuracy:  0.009999999776482582

MCP [후] -> modelCheckPoint 지점 
loss:  10.154399871826172
accuracy:  0.009999999776482582
'''