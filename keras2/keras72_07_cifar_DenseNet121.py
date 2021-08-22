from tensorflow.keras.layers import Dense, Flatten, GlobalAvgPool1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16, VGG19, ResNet50, ResNet101, InceptionResNetV2, DenseNet121
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


den = DenseNet121(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

den.trainable =False   

model = Sequential()
model.add(den)
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
    loss:  3.473520040512085
    accuracy:  0.16089999675750732
    걸린 시간:  1392.470942735672

    True False 
    loss:  3.5236217975616455
    accuracy:  0.0869000032544136
    걸린 시간:  77.8494598865509

    False False 
    loss:  3.489985466003418
    accuracy:  0.10639999806880951
    걸린 시간:  77.49098658561707

    False True 
    loss:  3.297816514968872
    accuracy:  0.10140000283718109
    걸린 시간:  1387.1144683361053

2)Flatten

    True True 
    loss:  7.148913860321045
    accuracy:  0.10000000149011612
    걸린 시간:  1389.3791062831879

    True False 
    loss:  3.4350523948669434
    accuracy:  0.09730000048875809
    걸린 시간:  81.19321846961975

    False False 
    loss:  2.6083879470825195
    accuracy:  0.0966000035405159
    걸린 시간:  77.51650714874268

    False True 
    loss:  2.6083879470825195
    accuracy:  0.0966000035405159
    걸린 시간:  77.51650714874268

2. cifar 100
1) GAP

    True True 
    loss:  5.403738498687744
    accuracy:  0.010900000110268593
    걸린 시간:  1389.757112979889

    True False 
    loss:  6.0106201171875
    accuracy:  0.010300000198185444
    걸린 시간:  77.22849297523499

    False False 
    loss:  5.652259349822998
    accuracy:  0.00800000037997961
    걸린 시간:  77.8190598487854

    False True 
    loss:  5.460933208465576
    accuracy:  0.01679999940097332
    걸린 시간:  1392.6751182079315

2)Flatten

    True True 
    loss:  30.400400161743164
    accuracy:  0.009999999776482582
    걸린 시간:  1390.0777661800385

    True False 
    loss:  5.799483776092529
    accuracy:  0.008999999612569809
    걸린 시간:  78.2096905708313

    False False 
    loss:  5.976329326629639
    accuracy:  0.011500000022351742
    걸린 시간:  78.31431555747986

    False True 
    loss:  6.668507099151611
    accuracy:  0.011099999770522118
    걸린 시간:  1392.3767549991608





'''

