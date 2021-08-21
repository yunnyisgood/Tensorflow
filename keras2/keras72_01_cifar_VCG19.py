from tensorflow.keras.layers import Dense, Flatten, GlobalAvgPool1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16, VGG19
import pandas as pd
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, QuantileTransformer, StandardScaler
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.python.keras.layers.pooling import GlobalAveragePooling2D, MaxPool2D, MaxPooling2D
import time
from tensorflow.keras.callbacks import EarlyStopping

# cifar10 실습 [FC를 모델로 하고 MAPooling 사용 ]
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# (x_train, y_train), (x_test, y_test) = cifar100.load_data()

x_train = x_train.reshape(50000, 32 * 32 * 3)
x_test = x_test.reshape(10000, 32 * 32 * 3)

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

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


vgg19 = VGG19(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# vgg16.trainable =True  # vgg훈련을 동결한다 -> 0이 된다 
vgg19.trainable =True  # vgg훈련을 동결한다 -> 0이 된다 

model = Sequential()
model.add(vgg19)
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

model.trainable = True

print(len(model.weights)) # 26 -> 30  -> 4개 증가 : layer 2개 증가 했기 때문에 총 layer, bias  각각 총 4개 증가 
print(len(model.trainable_weights))  # 0 -> 4 : 위와 동일           

# model.trainable=True # 전체 모델 훈련을 동결한다 
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
start_time = time.time()
es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1)
model.fit(x_train, y_train, verbose=1, epochs=100, batch_size=1024, validation_split=0.2, callbacks=es) 

# evaluate 
loss = model.evaluate(x_test, y_test)
print('loss: ', loss[0])
print('accuracy: ', loss[1])
print("걸린 시간: ", time.time()-start_time)


'''
1. cifar 10
1) GAP
    True True 
    loss:  1.8211287260055542
    accuracy:  0.24699999392032623
    걸린 시간:  1435.0262472629547

    True False 
    loss:  2.583721399307251
    accuracy:  0.09920000284910202

    False False 
    loss:  2.5722861289978027
    accuracy:  0.10090000182390213
    걸린 시간:  180.65854239463806

    False True 
    loss:  1.825405240058899
    accuracy:  0.24140000343322754
    걸린 시간:  1438.06161236763

2)Flatten

    True True 


    True False 


    False False 


    False True 

2. cifar 100
1) GAP

    True True 


    True False 


    False False 


    False True 

2)Flatten

    True True 


    True False 


    False False 


    False True 






'''

