from tensorflow.keras.layers import Dense, Flatten, GlobalAvgPool1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16, VGG19, ResNet50
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

print(x_train.shape, x_test.shape) 

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(50000, 32 , 32, 3)
x_test = x_test.reshape(10000, 32, 32, 3)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


res50 = ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

res50.trainable =False   

model = Sequential()
model.add(res50)
model.add(GlobalAveragePooling2D())
# model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

model.trainable = True

print(len(model.weights)) # 26 -> 30  -> 4개 증가 : layer 2개 증가 했기 때문에 총 layer, bias  각각 총 4개 증가 
print(len(model.trainable_weights))  # 0 -> 4 : 위와 동일           

# model.trainable=True # 전체 모델 훈련을 동결한다 
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
start_time = time.time()
es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=1)
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
    loss:  4.037076473236084
    accuracy:  0.11980000138282776
    걸린 시간:  34.174720287323

    True False 


    False False 


    False True 
    loss:  3.6683311462402344
    accuracy:  0.10080000013113022
    걸린 시간:  34.01014542579651

2)Flatten

    True True 
    loss:  3.458590269088745
    accuracy:  0.10350000113248825
    걸린 시간:  50.69389200210571

    True False 


    False False 


    False True 


'''