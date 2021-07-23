'''
overfit을 극복하자
1. 전체 훈련 데이터가 많이 있을수록 좋다
2. 정규화(normarlization) 고
-> 전처리 할 때 scailing에서 했던 정규화로는 부족
-> layer에서 layer로 넘어갈 때 활성화 함수로 감쌀때 정규화 시켜줘야 한다는 의미
-> layer별로 정규화를 시켜주는것은 어떠냐는 의미 
3. dropout
: 각 layer마다 몇개의 node를 빼고 계산했을 때 과적합이 줄어듦었음을 알 수 있다
: 이 때 머신은 랜덤하게 제외시킬 노트를 일정비율로 솎아준다.
'''

import numpy as np
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, PowerTransformer, QuantileTransformer
import time 
import matplotlib.pyplot as plt


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

x_train = x_train.reshape(50000, 32 , 32, 3)
x_test = x_test.reshape(10000, 32, 32, 3)


# 전처리 하기 -> one-hot-encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train.shape, y_test.shape)


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

model.add(Flatten()) 
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dense(100, activation='softmax'))


# compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=100, verbose=1, callbacks=[es], validation_split=0.2,
shuffle=True, batch_size=256)
end_time = time.time() - start_time



# evaluate, predict
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

dropout 실행
걸린시간:  603.1085088253021
category:  2.0480213165283203
accuracy:  0.4699999988079071


'''